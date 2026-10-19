"""Session-owned news exploration using the existing bounded selector.

Each map keeps its own search snapshots and cursors. External calls happen
outside revision transactions; stale results cannot overwrite newer state.
"""
from __future__ import annotations

import time
import uuid
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app import store
from app.agents import news_fetcher
from app.agents.article_embeddings import EmbeddingUnavailable
from app.agents.decision_client import DecisionUnavailable
from app.agents.naver_client import NaverNewsError
from app.agents.news_map import NewsMapResult, build_news_map
from app.agents.related_selector import article_id
from app.config import settings
from app.utils import cache_articles
from app.v2 import state
from app.v2.errors import V2Error

PUBLIC_FIELDS = ('map_id', 'revision', 'query', 'state', 'root_node_id', 'nodes', 'branches',
                 'initial_selection', 'selected_news_ids', 'detail_news_id', 'report_panel_open',
                 'active_report_job_id', 'last_result', 'expires_at')
ARTICLE_FIELDS = ('news_id', 'title', 'description', 'source', 'published_at', 'url', 'naver_url',
                  'thumbnail_url', 'keywords', 'categories', '_search_rank', '_search_end',
                  '_search_keyword', '_searched_at')


class Input(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class MapCreate(Input):
    query: str = Field(min_length=1, max_length=100)

    @field_validator('query', mode='before')
    @classmethod
    def strip(cls, value):
        return value.strip() if isinstance(value, str) else value


class Revision(Input):
    expected_revision: int = Field(ge=1)


class MapUpdate(Revision):
    selected_news_ids: list[str] | None = Field(default=None, max_length=10)
    detail_news_id: str | None = None
    report_panel_open: bool | None = None

    @field_validator('selected_news_ids')
    @classmethod
    def unique(cls, value):
        if value is not None and len(value) != len(set(value)):
            raise ValueError('기사 ID는 중복될 수 없습니다.')
        return value


class BranchRequest(Revision):
    parent_node_id: str = Field(min_length=1, max_length=100)
    mode: Literal['expand', 'research']


def card(article: dict) -> dict:
    source = article.get('source') or ''
    return {'news_id': article_id(article), 'title': article.get('title') or '',
            'description': article.get('description') or '',
            'source_name': source.get('name', '') if isinstance(source, dict) else source,
            'published_at': article.get('published_at') or None,
            'source_url': article.get('naver_url') or article.get('url') or None,
            'thumbnail_url': article.get('thumbnail_url') or None,
            'keywords': article.get('keywords') or [], 'categories': article.get('categories') or []}


def snapshot(article: dict) -> dict:
    result = {k: deepcopy(article[k]) for k in ARTICLE_FIELDS if k in article}
    result['news_id'] = article_id(article)
    return result


def public_map(value: dict) -> dict:
    return {k: deepcopy(value.get(k)) for k in PUBLIC_FIELDS}


def selection(result: NewsMapResult) -> dict:
    count = len(result.items)
    status = result.status if result.status != 'expandable' else 'insufficient'
    return {'status': status, 'reason': result.reason,
            'requested': result.target, 'returned': count, 'shortage': count < result.target,
            'budget_exhausted': result.reason in ('candidate_limit', 'search_limit', 'timeout', 'history_limit')}


def _shortage(target: int, reason: str) -> dict:
    return {'status': 'insufficient', 'reason': reason, 'requested': target, 'returned': 0,
            'shortage': True, 'budget_exhausted': reason == 'history_limit'}


def capabilities(session: dict) -> dict:
    return {'max_report_articles': 10 if session['prototype_tier'] == 'PAID' else 1,
            'initial_related_target': 4, 'branch_related_target': 3,
            'max_map_nodes': settings.v2_max_map_nodes, 'max_map_depth': settings.v2_max_map_depth}


async def load_map(sid: str, map_id: str) -> dict:
    result = await state.resource(sid, 'map', map_id)
    result['expires_at'] = (await state.session(sid))['expires_at']
    return result


async def commit_map(sid: str, value: dict, expected_revision: int) -> dict:
    session = await state.session(sid)
    value = deepcopy(value)
    value['revision'] = expected_revision + 1
    value['expires_at'] = session['expires_at']
    return await state.cas(state.key(sid, 'map', value['map_id']), expected_revision, value,
                           state.epoch(session['expires_at']) - time.time())


def check_revision(value: dict, expected: int) -> None:
    if value['revision'] != expected:
        raise V2Error(409, 'MAP_REVISION_CONFLICT', '다른 요청에서 탐색이 변경되었습니다.',
                      details={'revision': value['revision']})


async def _select(center: dict, pool: list[dict], target: int, excluded: list[dict], context: dict):
    try:
        return await build_news_map(center, target=target, pool=pool, exclude_articles=excluded,
                                    search_context=context, expand=True)
    except (EmbeddingUnavailable, DecisionUnavailable) as exc:
        raise V2Error(503, 'DEPENDENCY_UNAVAILABLE', '연관 기사 판단 서비스를 사용할 수 없습니다.', True) from exc
    except NaverNewsError as exc:
        raise V2Error(502, 'SEARCH_FAILED', '기사 검색을 완료하지 못했습니다.', True) from exc


def _node(value: dict, article: dict, parent: str | None, level: int, batch: int = 0) -> dict:
    node_id = f"n{value['next_node_seq']}"
    value['next_node_seq'] += 1
    value.setdefault('articles', {})[article_id(article)] = snapshot(article)
    return {'node_id': node_id, 'news_id': article_id(article), 'parent_node_id': parent,
            'level': level, 'expanded': False, 'batch': batch, 'news': card(article)}


async def create_map(sid: str, query: str) -> dict:
    session = await state.session(sid, touch=True)
    async with state.mutex(f"v2:map-create:{sid}"):
        live = [rid for rid in await state.keys(sid, 'map') if await state.get(state.key(sid, 'map', rid))]
        if len(live) >= settings.v2_max_maps:
            raise V2Error(422, 'SESSION_MAP_LIMIT_REACHED', '열려 있는 탐색이 너무 많습니다.')
        try:
            page = await news_fetcher.fetch_search_page(query, start=1, size=settings.news_map_initial_candidates)
            articles = await news_fetcher.inspect_search_articles(page.articles)
            observed = state.iso()
            articles = await cache_articles([{**a, '_search_keyword': query, '_search_end': page.end,
                                               '_searched_at': observed} for a in articles])
        except NaverNewsError as exc:
            raise V2Error(502, 'SEARCH_FAILED', '기사 검색을 완료하지 못했습니다.', True) from exc
        except RuntimeError as exc:
            raise V2Error(503, 'DEPENDENCY_UNAVAILABLE', 'v2 뉴스 공급원이 설정되지 않았습니다.', True) from exc
        value = {'map_id': str(uuid.uuid4()), 'revision': 1, 'query': query,
                 'state': 'active' if articles else 'empty', 'root_node_id': None,
                 'nodes': [], 'branches': [], 'selected_news_ids': [], 'detail_news_id': None,
                 'report_panel_open': False, 'active_report_job_id': None, 'last_result': None,
                 'expires_at': session['expires_at'], 'next_node_seq': 1, 'articles': {},
                 'candidate_articles': [snapshot(a) for a in articles], 'branch_history': {},
                 'search_context': {'query': query, 'sort': 'relevance', 'observed_at': observed,
                                    'raw_start': page.start, 'raw_end': page.end,
                                    'candidate_news_ids': [article_id(a) for a in articles],
                                    'selector': settings.news_map_selector,
                                    'next_positions': {query: page.end + 1}}}
        if articles:
            root = _node(value, articles[0], None, 0)
            result = await _select(snapshot(articles[0]), articles, 4, [], value['search_context'])
            children = [_node(value, item.article, root['node_id'], 1) for item in result.items]
            root['expanded'] = True
            value.update(root_node_id=root['node_id'], nodes=[root, *children], initial_selection=selection(result))
            value['branches'] = [{'parent_node_id': root['node_id'],
                                  'child_node_ids': [n['node_id'] for n in children],
                                  'display_selection': selection(result), 'last_attempt': selection(result)}]
            value['branch_history'][root['node_id']] = {
                'previously_displayed_news_ids': [n['news_id'] for n in children],
                'previous_articles': [snapshot(item.article) for item in result.items],
                'search_context': deepcopy(value['search_context'])}
        else:
            value['initial_selection'] = _shortage(4, 'no_search_results')
        value['initial_snapshot'] = deepcopy({k: value[k] for k in ('root_node_id', 'nodes', 'branches',
                                                                  'initial_selection', 'branch_history')})
        # A session might have been deleted while search was awaiting an API.
        session = await state.session(sid)
        value['expires_at'] = session['expires_at']
        await state.put(state.key(sid, 'map', value['map_id']), value,
                        state.epoch(session['expires_at']) - time.time())
        await state.register_resource(sid, 'map', value['map_id'], value['expires_at'])
        return value


async def cancel_detached(sid: str, job_id: str | None) -> None:
    if job_id:
        from app.v2.reports import cancel_job
        try:
            await cancel_job(sid, job_id)
        except V2Error as exc:
            if exc.status not in (404, 410):
                raise


async def update_map(sid: str, map_id: str, body: MapUpdate) -> dict:
    value = await load_map(sid, map_id)
    check_revision(value, body.expected_revision)
    visible = {node['news_id'] for node in value['nodes']}
    fields = body.model_fields_set
    old_job = None
    if 'selected_news_ids' in fields:
        selected = body.selected_news_ids
        if selected is None:
            raise V2Error(422, 'INVALID_REQUEST', '기사 선택 목록은 배열이어야 합니다.')
        if any(nid not in visible for nid in selected):
            raise V2Error(422, 'INVALID_REQUEST', '현재 맵에 없는 기사를 선택할 수 없습니다.')
        session = await state.session(sid)
        if len(selected) > capabilities(session)['max_report_articles']:
            raise V2Error(403, 'PLAN_ARTICLE_LIMIT_EXCEEDED', '무료 리포트는 기사 1건을 선택할 수 있습니다.')
        if set(selected) != set(value['selected_news_ids']):
            old_job, value['active_report_job_id'] = value.get('active_report_job_id'), None
            value['last_result'] = None
        value['selected_news_ids'] = selected
        if not selected:
            value['report_panel_open'] = False
    if 'detail_news_id' in fields:
        if body.detail_news_id is not None and body.detail_news_id not in visible:
            raise V2Error(422, 'INVALID_REQUEST', '현재 맵에 없는 상세 기사입니다.')
        value['detail_news_id'] = body.detail_news_id
    if 'report_panel_open' in fields:
        if body.report_panel_open is None:
            raise V2Error(422, 'INVALID_REQUEST', '패널 상태는 boolean이어야 합니다.')
        value['report_panel_open'] = body.report_panel_open
    value = await commit_map(sid, value, body.expected_revision)
    await cancel_detached(sid, old_job)
    return value


def _descendants(value: dict, parent: str) -> set[str]:
    removed: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in value['nodes']:
            if node['parent_node_id'] in ({parent} | removed) and node['node_id'] not in removed:
                removed.add(node['node_id'])
                changed = True
    return removed


async def branch_map(sid: str, map_id: str, body: BranchRequest) -> dict:
    value = await load_map(sid, map_id)
    check_revision(value, body.expected_revision)
    parent = next((n for n in value['nodes'] if n['node_id'] == body.parent_node_id), None)
    if parent is None:
        raise V2Error(422, 'INVALID_REQUEST', '현재 맵에 없는 부모 노드입니다.')
    if (body.mode == 'expand') == parent['expanded']:
        raise V2Error(422, 'INVALID_REQUEST', '펼친 노드는 재검색을 사용해야 합니다.')
    if parent['level'] >= settings.v2_max_map_depth:
        raise V2Error(422, 'MAP_LIMIT_REACHED', '탐색 깊이 한도에 도달했습니다.')
    target = 4 if parent['level'] == 0 else 3
    history = value['branch_history'].setdefault(parent['node_id'], {
        'previously_displayed_news_ids': [], 'previous_articles': [],
        'search_context': deepcopy(value['search_context'])})
    removed = _descendants(value, parent['node_id']) if body.mode == 'research' else set()
    if len(history['previously_displayed_news_ids']) >= settings.v2_branch_history_limit:
        attempt = _shortage(target, 'history_limit')
        items = []
    else:
        excluded = [value['articles'][n['news_id']] for n in value['nodes']]
        excluded += history['previous_articles']
        context = deepcopy(history['search_context'])
        result = await _select(value['articles'][parent['news_id']], value['candidate_articles'],
                               min(target, settings.v2_branch_history_limit
                                   - len(history['previously_displayed_news_ids'])), excluded, context)
        items, attempt = result.items, selection(result)
        attempt['requested'] = target
        attempt['shortage'] = len(items) < target
        history['search_context'] = context
    if len(value['nodes']) - len(removed) + len(items) > settings.v2_max_map_nodes:
        raise V2Error(422, 'MAP_LIMIT_REACHED', '탐색 기사 수 한도에 도달했습니다.')
    previous = next((b for b in value['branches'] if b['parent_node_id'] == parent['node_id']), None)
    old_job = None
    if items:
        value['nodes'] = [n for n in value['nodes'] if n['node_id'] not in removed]
        value['branches'] = [b for b in value['branches'] if b['parent_node_id'] not in removed
                             and b['parent_node_id'] != parent['node_id']]
        for rid in removed:
            value['branch_history'].pop(rid, None)
        batch = max((n['batch'] for n in value['nodes']), default=0) + 1
        children = [_node(value, item.article, parent['node_id'], parent['level'] + 1, batch) for item in items]
        value['nodes'].extend(children)
        previous = {'parent_node_id': parent['node_id'], 'child_node_ids': [n['node_id'] for n in children],
                    'display_selection': attempt, 'last_attempt': attempt}
        value['branches'].append(previous)
        visible = {n['news_id'] for n in value['nodes']}
        selected = [nid for nid in value['selected_news_ids'] if nid in visible]
        if selected != value['selected_news_ids']:
            old_job, value['active_report_job_id'] = value.get('active_report_job_id'), None
            value['selected_news_ids'], value['last_result'] = selected, None
            if not selected:
                value['report_panel_open'] = False
        if value['detail_news_id'] not in visible:
            value['detail_news_id'] = None
        history['previously_displayed_news_ids'].extend(n['news_id'] for n in children)
        history['previous_articles'].extend(snapshot(item.article) for item in items)
    elif previous:
        previous['last_attempt'] = attempt  # Zero results preserve valid existing children.
    else:
        value['branches'].append({'parent_node_id': parent['node_id'], 'child_node_ids': [],
                                  'display_selection': attempt, 'last_attempt': attempt})
    parent['expanded'] = True
    value = await commit_map(sid, value, body.expected_revision)
    await cancel_detached(sid, old_job)
    return value


async def reset_map(sid: str, map_id: str, expected: int) -> dict:
    value = await load_map(sid, map_id)
    check_revision(value, expected)
    old_job = value.get('active_report_job_id')
    initial = deepcopy(value['initial_snapshot'])
    remap = {}
    for node in initial['nodes']:
        remap[node['node_id']] = f"n{value['next_node_seq']}"
        value['next_node_seq'] += 1
    for node in initial['nodes']:
        node['node_id'] = remap[node['node_id']]
        node['parent_node_id'] = remap.get(node['parent_node_id'])
    for branch in initial['branches']:
        branch['parent_node_id'] = remap[branch['parent_node_id']]
        branch['child_node_ids'] = [remap[rid] for rid in branch['child_node_ids']]
    initial['branch_history'] = {remap[rid]: h for rid, h in initial['branch_history'].items()}
    initial['root_node_id'] = remap.get(initial['root_node_id'])
    value.update(initial)
    value.update(selected_news_ids=[], detail_news_id=None, report_panel_open=False,
                 active_report_job_id=None, last_result=None)
    value = await commit_map(sid, value, expected)
    await cancel_detached(sid, old_job)
    return value


async def delete_map(sid: str, map_id: str) -> None:
    value = await load_map(sid, map_id)
    await state.tombstone_map(sid, map_id)
    await cancel_detached(sid, value.get('active_report_job_id'))


async def news_card(sid: str, news_id: str) -> dict:
    await state.session(sid)
    article = await store.get_news(news_id)
    if not article:
        raise V2Error(404, 'RESOURCE_NOT_FOUND', '찾을 수 없는 기사입니다.')
    return card(article)
