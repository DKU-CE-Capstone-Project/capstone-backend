"""v2 contract/security tests with isolated state and deterministic candidates."""
import asyncio
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from app import session as base_session
from app.agents import related_selector
from app.agents.decision_client import DecisionClient, normalize
from app.agents.decision_policy import PAIR_QUESTIONS
from app.agents.naver_client import NaverNewsError, NaverNewsPage
from app.agents.news_map import NewsMapResult, build_news_map
from app.agents.related_selector import RankedArticle, identity_keys
from app.config import settings
from app.main import app
from app.v2 import maps, state
from tests.test_decision_news_map import article
from tests.test_decision_news_map import response as decision_response


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.setattr(settings, 'v2_state_memory', True)
    monkeypatch.setattr(settings, 'session_cookie_secure', False)
    monkeypatch.setattr(settings, 'prototype_tier', 'FREE')
    state._memory.clear()
    articles = [{'news_id': f'a{i}', 'title': f'기사 {i}', 'description': f'독립 근거 {i}',
                 'url': f'https://fixture.invalid/{i}', 'source': '테스트', 'keywords': []}
                for i in range(12)]
    calls = []

    async def fetch(query, **kwargs):
        return NaverNewsPage(articles=deepcopy(articles), total=len(articles), start=1, end=12)

    async def select(center, *, pool, target, exclude_articles, search_context, expand, find_issues=False):
        excluded = set().union(*(identity_keys(a) for a in [center, *exclude_articles]))
        eligible = [a for a in pool if not identity_keys(a) & excluded]
        chosen = eligible[:target]
        calls.append({'target': target, 'center': center, 'excluded': excluded, 'context': deepcopy(search_context)})
        return NewsMapResult([RankedArticle(a, 0.9, 0.9) for a in chosen],
                             'complete' if len(chosen) == target else 'insufficient', None, target)

    monkeypatch.setattr(maps.news_fetcher, 'fetch_search_page', fetch)
    monkeypatch.setattr(maps, 'build_news_map', select)
    with TestClient(app) as client:
        response = client.get('/api/v2/session')
        assert response.status_code == 200
        yield client, articles, calls, response.json()
    state._memory.clear()


def create(client, query='검색'):
    response = client.post('/api/v2/maps', json={'query': query})
    assert response.status_code == 201, response.text
    return response.json()


def test_session_secret_and_capabilities(fixture):
    client, _, _, session = fixture
    assert session['capabilities']['max_report_articles'] == 1
    assert session['capabilities']['initial_related_target'] == 4
    assert session['resumed'] is False
    assert 'sid' not in session and 'session_id' not in session
    assert client.get('/api/v2/session').json()['resumed'] is True
    assert 'httponly' in client.get('/api/v2/session').headers['set-cookie'].lower()


def test_initial_four_and_map_private_fields(fixture):
    client, _, calls, _ = fixture
    value = create(client)
    assert len(value['nodes']) == 5 and calls[-1]['target'] == 4
    assert set(value) == set(maps.PUBLIC_FIELDS)
    assert 'articles' not in value and 'search_context' not in value
    assert value['initial_selection']['returned'] == 4
    assert value['next_issue'] is None
    assert value['can_find_next_issue'] is True


def issue_search_fixture(monkeypatch, *, related=0, fail_at=None, search_budget=3,
                         first_related=(), second_related=('second-background',),
                         second_rank=7, distinct_issues=('second', 'third')):
    monkeypatch.setattr(settings, 'news_map_selector', 'decision')
    monkeypatch.setattr(settings, 'news_map_supplement_max_searches', search_budget)
    monkeypatch.setattr(maps, 'build_news_map', build_news_map)
    titles = {1: 'first', 2: 'first-copy', 3: 'unclear', 4: 'first-angle-1', 5: 'first-angle-2',
              second_rank: 'second', 8: 'first-copy-late', 41: 'third', 50: 'second-background', 51: 'outside'}
    articles = [article(titles.get(rank, f'copy-{rank}'), rank) for rank in range(1, 52)]
    searched, compared = [], []

    async def fetch(query, *, start, size):
        searched.append((query, start, size))
        if start == fail_at:
            raise NaverNewsError('fixture')
        items = articles[start - 1:start - 1 + size]
        return NaverNewsPage(deepcopy(items), len(articles), start, start + len(items) - 1)

    async def ask(self, center, candidate, questions):
        a, b = center['news_id'], candidate['news_id']
        compared.append((a, b))
        if questions is PAIR_QUESTIONS:
            return normalize(decision_response(questions, redundancy='distinct'), questions)
        occurrence, contribution, helpfulness = 'same', 'repeat', '0'
        if a == 'first' and (b in first_related or (b.startswith('first-angle') and related)):
            contribution, helpfulness = 'detail', '3'
        elif a == 'first' and b == 'unclear':
            occurrence, contribution = 'unclear', 'unrelated'
        elif a == 'second' and b in second_related:
            occurrence, contribution, helpfulness = 'different', 'background', '3'
        elif b in distinct_issues or a == 'second':
            occurrence, contribution = 'different', 'tangential'
        return normalize(decision_response(questions, occurrence=occurrence, contribution=contribution,
                                           helpfulness=helpfulness), questions)

    monkeypatch.setattr(maps.news_fetcher, 'fetch_search_page', fetch)
    monkeypatch.setattr(DecisionClient, 'ask', ask)
    return searched, compared


@pytest.mark.parametrize('second_rank', [7, 48])
def test_complete_map_can_open_another_issue_without_eager_expansion(fixture, monkeypatch, second_rank):
    client, _, _, _ = fixture
    second_related = ('copy-14', 'copy-15', 'copy-16', 'copy-17')
    searched, compared = issue_search_fixture(
        monkeypatch, first_related=('copy-10', 'copy-11', 'copy-12', 'copy-13'),
        second_related=second_related, second_rank=second_rank, distinct_issues=('second',),
    )
    first = create(client, '반도체')
    assert first['initial_selection']['returned'] == 4
    assert first['can_find_next_issue'] is True
    assert first['next_issue'] == ({'news_id': 'second', 'title': 'second'} if second_rank == 7 else None)
    assert searched == [('반도체', 1, 20)]
    first_comparisons = compared.count(('first', 'first-copy'))
    assert first_comparisons == 1 and not any(a == 'second' for a, _ in compared)

    path = f"/api/v2/maps/{first['map_id']}"
    response = client.post(f'{path}/next-issue')
    assert response.status_code == 200, response.text
    next_map = response.json()
    assert next_map['previous_issue_map_id'] == first['map_id']
    assert next_map['nodes'][0]['news_id'] == 'second'
    assert {n['news_id'] for n in next_map['nodes'][1:]} == set(second_related)
    assert sum(a == 'second' for a, _ in compared) == (19 if second_rank == 7 else 20)
    assert compared.count(('first', 'first-copy')) == first_comparisons
    assert searched[1:] == ([] if second_rank == 7 else [('반도체', 21, 20), ('반도체', 41, 10)])
    assert client.get(path).json()['nodes'] == first['nodes']


def test_no_other_issue_keeps_the_complete_map_and_its_selection(fixture, monkeypatch):
    client, _, _, _ = fixture
    searched, compared = issue_search_fixture(
        monkeypatch, first_related=('copy-10', 'copy-11', 'copy-12', 'copy-13'), distinct_issues=(),
    )
    first = create(client, '반도체')
    path = f"/api/v2/maps/{first['map_id']}"
    selected = client.patch(path, json={'expected_revision': 1, 'selected_news_ids': ['first']}).json()
    response = client.post(f'{path}/next-issue')
    assert response.status_code == 200, response.text
    after = response.json()
    assert after == {**selected, 'revision': selected['revision'] + 1, 'can_find_next_issue': False,
                     'expires_at': after['expires_at']}
    assert searched == [('반도체', 1, 20), ('반도체', 21, 20), ('반도체', 41, 10)]
    assert not any(b == 'outside' for _, b in compared)
    count = len(compared)
    assert client.post(f'{path}/next-issue').json()['revision'] == after['revision']
    assert len(compared) == count


@pytest.mark.parametrize('related', [0, 2])
def test_next_issue_reuses_the_completed_search_only_after_user_action(fixture, monkeypatch, related):
    client, _, _, _ = fixture
    searched, compared = issue_search_fixture(monkeypatch, related=related)
    monkeypatch.setattr(settings, 'v2_max_maps', 2)
    first = create(client, '반도체')
    assert first['state'] == 'active'
    assert first['initial_selection']['returned'] == related
    assert first['next_issue'] == {'news_id': 'second', 'title': 'second'}
    assert first['nodes'][0]['news_id'] == 'first'
    assert len(first['nodes']) == 1 + related
    assert {a for a, _ in compared if a not in ('first-angle-1', 'first-angle-2')} == {'first'}
    assert searched == [('반도체', 1, 20), ('반도체', 21, 20), ('반도체', 41, 10)]
    assert all(b != 'outside' for _, b in compared)

    path = f"/api/v2/maps/{first['map_id']}"
    assert client.patch(path, json={'expected_revision': 1, 'selected_news_ids': ['first']}).status_code == 200
    with TestClient(app) as other:
        other.get('/api/v2/session')
        assert other.post(f'{path}/next-issue').status_code == 404
    opened = client.post(f'{path}/next-issue')
    assert opened.status_code == 200, opened.text
    second = opened.json()
    assert second['query'] == first['query']
    assert second['previous_issue_map_id'] == first['map_id']
    assert [n['news_id'] for n in second['nodes']] == ['second', 'second-background']
    assert second['next_issue']['news_id'] == 'third'  # Skip a later repeat of the first issue.
    assert second['selected_news_ids'] == []
    assert client.get(path).json()['selected_news_ids'] == ['first']
    assert len(searched) == 3  # No NAVER re-search, including for the article at position 50.

    second_path = f"/api/v2/maps/{second['map_id']}"
    assert client.patch(second_path, json={'expected_revision': 1, 'selected_news_ids': ['second']}).status_code == 200
    compared_count = len(compared)
    reopened = client.post(f'{path}/next-issue').json()
    assert reopened['map_id'] == second['map_id'] and reopened['selected_news_ids'] == ['second']
    assert len(compared) == compared_count

    separate = create(client, '다른 검색')  # Issue views share a search slot.
    assert client.delete(f'{second_path}?include_issues=true').status_code == 204
    assert client.get(path).status_code == client.get(second_path).status_code == 410
    assert client.get(f"/api/v2/maps/{separate['map_id']}").status_code == 200


@pytest.mark.parametrize(('second_related', 'expected_evaluated'), [
    (('copy-10', 'copy-11', 'copy-12', 'copy-13'), 20),
    (('copy-22', 'copy-23', 'copy-24', 'copy-25'), 40),
    (('copy-22', 'copy-23', 'copy-24', 'second-background'), 49),
])
def test_next_issue_stops_evaluating_after_the_round_that_finds_four(
    fixture, monkeypatch, second_related, expected_evaluated,
):
    client, _, _, _ = fixture
    searched, compared = issue_search_fixture(monkeypatch, second_related=second_related)
    first = create(client, '반도체')
    response = client.post(f"/api/v2/maps/{first['map_id']}/next-issue")
    assert response.status_code == 200, response.text
    second = response.json()
    assert second['initial_selection']['status'] == 'complete'
    assert {n['news_id'] for n in second['nodes'][1:]} == set(second_related)
    assert sum(center == 'second' for center, _ in compared) == expected_evaluated
    assert searched == [('반도체', 1, 20), ('반도체', 21, 20), ('반도체', 41, 10)]


@pytest.mark.parametrize(('fail_at', 'budget', 'status'), [(41, 3, 'partial'), (None, 1, 'insufficient')])
def test_next_issue_is_not_offered_when_the_search_stops_early(fixture, monkeypatch, fail_at, budget, status):
    client, _, _, _ = fixture
    issue_search_fixture(monkeypatch, fail_at=fail_at, search_budget=budget)
    first = create(client)
    assert first['initial_selection']['status'] == status
    assert first['next_issue'] is None
    denied = client.post(f"/api/v2/maps/{first['map_id']}/next-issue")
    assert denied.status_code == 422
    assert denied.json()['error']['code'] == 'NEXT_ISSUE_UNAVAILABLE'


def test_embedding_mode_can_offer_a_distinct_issue_when_naver_has_fewer_than_fifty(fixture, monkeypatch):
    client, _, _, _ = fixture
    monkeypatch.setattr(maps, 'build_news_map', build_news_map)
    articles = [
        article('first', 1, title='반도체 신규 생산시설 착공', description='용인 공장 증설'),
        article('repeat', 2, title='반도체 신규 생산시설 착공', description='용인 공장 증설'),
        article('second', 3, title='유럽 법원, 칩 설계 특허 침해 첫 심리', description='반도체 설계도 무단 이용 쟁점'),
    ]
    searched = []

    async def fetch(query, *, start, size):
        searched.append((start, size))
        return NaverNewsPage(deepcopy(articles), len(articles), start, len(articles))

    async def vector(article):
        return [0.0, 1.0] if article['news_id'] == 'second' else [1.0, 0.0]

    monkeypatch.setattr(maps.news_fetcher, 'fetch_search_page', fetch)
    monkeypatch.setattr(related_selector, 'article_vector', vector)
    first = create(client, '반도체')
    assert first['initial_selection']['returned'] == 0
    assert first['next_issue']['news_id'] == 'second'
    second = client.post(f"/api/v2/maps/{first['map_id']}/next-issue").json()
    assert [node['news_id'] for node in second['nodes']] == ['second']
    assert second['next_issue'] is None
    assert searched == [(1, 20)]


def test_free_selection_limit_and_revision_conflict(fixture):
    client, _, _, _ = fixture
    value = create(client)
    url = f"/api/v2/maps/{value['map_id']}"
    assert client.patch(url, json={'expected_revision': 1, 'selected_news_ids': ['a0', 'a1']}).status_code == 403
    changed = client.patch(url, json={'expected_revision': 1, 'selected_news_ids': ['a1']})
    assert changed.status_code == 200 and changed.json()['revision'] == 2
    stale = client.patch(url, json={'expected_revision': 1, 'selected_news_ids': []})
    assert stale.status_code == 409
    assert stale.json()['error']['details']['revision'] == 2


def test_paid_capability_requires_server_opt_in(fixture, monkeypatch):
    client, _, _, _ = fixture
    monkeypatch.setattr(settings, 'prototype_tier', 'PAID')
    monkeypatch.setattr(settings, 'paid_demo_enabled', False)
    with TestClient(app) as other:
        assert other.get('/api/v2/session').json()['prototype_tier'] == 'FREE'
    monkeypatch.setattr(settings, 'paid_demo_enabled', True)
    with TestClient(app) as other:
        assert other.get('/api/v2/session').json()['capabilities']['max_report_articles'] == 10
    assert client.get('/api/v2/session').json()['prototype_tier'] == 'FREE'


def test_other_session_map_and_expired_session(fixture):
    client, _, _, _ = fixture
    value = create(client)
    url = f"/api/v2/maps/{value['map_id']}"
    with TestClient(app) as other:
        other.get('/api/v2/session')
        assert other.get(url).status_code == 404
    sid = client.cookies.get('econmind_sid')
    base_session._memory_store.pop(sid)
    response = client.get(url)
    assert response.status_code == 410 and response.json()['error']['code'] == 'SESSION_EXPIRED'
    assert 'set-cookie' not in response.headers
    assert client.get('/api/v2/session').json()['resumed'] is False


def test_map_restore_and_reset_use_new_node_ids(fixture):
    client, _, _, _ = fixture
    value = create(client)
    old_ids = {n['node_id'] for n in value['nodes']}
    url = f"/api/v2/maps/{value['map_id']}"
    assert client.get(url).json()['nodes'] == value['nodes']
    reset = client.post(url + '/reset', json={'expected_revision': 1}).json()
    assert reset['revision'] == 2
    assert old_ids.isdisjoint({n['node_id'] for n in reset['nodes']})
    assert [n['news_id'] for n in reset['nodes']] == [n['news_id'] for n in value['nodes']]


def test_branch_three_excludes_all_visible_and_research_history(fixture):
    client, _, calls, _ = fixture
    value = create(client)
    node = value['nodes'][1]
    url = f"/api/v2/maps/{value['map_id']}/branches"
    expanded = client.post(url, json={'expected_revision': 1, 'parent_node_id': node['node_id'], 'mode': 'expand'})
    assert expanded.status_code == 200
    after = expanded.json()
    assert len(after['nodes']) == 8 and calls[-1]['target'] == 3
    assert {f"id:a{i}" for i in range(5)} <= calls[-1]['excluded']
    researched = client.post(url, json={'expected_revision': 2, 'parent_node_id': node['node_id'], 'mode': 'research'})
    assert researched.status_code == 200
    assert {f"id:a{i}" for i in range(8)} <= calls[-1]['excluded']
    assert len({n['news_id'] for n in researched.json()['nodes']}) == len(researched.json()['nodes'])


def test_zero_research_retains_tree(fixture):
    client, articles, _, _ = fixture
    del articles[5:]
    value = create(client)
    response = client.post(f"/api/v2/maps/{value['map_id']}/branches", json={
        'expected_revision': 1, 'parent_node_id': value['root_node_id'], 'mode': 'research'})
    assert response.status_code == 200
    after = response.json()
    assert after['nodes'] == value['nodes']
    assert after['branches'][0]['display_selection']['returned'] == 4
    assert after['branches'][0]['last_attempt']['returned'] == 0


def test_empty_map_is_normal_and_invalid_query_is_error(fixture):
    client, articles, _, _ = fixture
    articles.clear()
    value = create(client)
    assert value['state'] == 'empty' and value['nodes'] == []
    assert value['initial_selection']['reason'] == 'no_search_results'
    assert client.post('/api/v2/maps', json={'query': '   '}).status_code == 422


def test_foreign_origin_prevents_changes_and_delete_clears_access(fixture):
    client, _, _, _ = fixture
    value = create(client)
    url = f"/api/v2/maps/{value['map_id']}"
    response = client.patch(url, headers={'Origin': 'https://foreign.invalid'},
                            json={'expected_revision': 1, 'selected_news_ids': ['a0']})
    assert response.status_code == 403 and client.get(url).json()['revision'] == 1
    assert client.delete('/api/v2/session').status_code == 204
    assert client.get(url).status_code == 410


def test_map_limit_and_own_resource_tombstone(fixture, monkeypatch):
    client, _, _, _ = fixture
    monkeypatch.setattr(settings, 'v2_max_maps', 1)
    value = create(client)
    assert client.post('/api/v2/maps', json={'query': '추가'}).json()['error']['code'] == 'SESSION_MAP_LIMIT_REACHED'
    url = f"/api/v2/maps/{value['map_id']}"
    sid = client.cookies.get('econmind_sid')
    assert client.delete(url).status_code == 204
    deleted = client.get(url)
    assert deleted.status_code == 410 and deleted.json()['error']['code'] == 'RESOURCE_EXPIRED'
    with TestClient(app) as other:
        other.get('/api/v2/session')
        assert other.get(url).status_code == 404
    assert asyncio.run(state.keys(sid, 'map')) == []

    renewed = []
    original_expire = state._expire

    async def record_expire(storage_key, ttl):
        renewed.append(storage_key)
        await original_expire(storage_key, ttl)

    monkeypatch.setattr(state, '_expire', record_expire)
    assert client.get('/api/v2/session').status_code == 200
    assert state.key(sid, 'map', value['map_id']) not in renewed
    assert client.post('/api/v2/maps', json={'query': '대체'}).status_code == 201


def test_strict_revision_and_documented_article_limit_error(fixture):
    import uuid
    client, _, _, _ = fixture
    value = create(client)
    assert client.patch(f"/api/v2/maps/{value['map_id']}",
                        json={'expected_revision': True, 'selected_news_ids': []}).status_code == 422
    over = client.post('/api/v2/report-jobs', headers={'Idempotency-Key': str(uuid.uuid4())}, json={
        'map_id': value['map_id'], 'expected_revision': 1, 'client_request_id': str(uuid.uuid4()),
        'news_ids': [f'a{i}' for i in range(11)]})
    assert over.status_code == 422 and over.json()['error']['code'] == 'ARTICLE_LIMIT_EXCEEDED'


@pytest.mark.asyncio
async def test_v2_real_decision_selector_honors_target_and_excluded_identity(monkeypatch):
    from pydantic import SecretStr

    from app.agents import news_map
    from tests.test_decision_news_map import article, mock_http
    monkeypatch.setattr(settings, 'news_map_selector', 'decision')
    monkeypatch.setattr(settings, 'openai_api_key', SecretStr('local-fixture-only'))
    calls = mock_http(monkeypatch)
    center = article('center')
    pool = [center, article('exclude'), *[article(f'background{i}', i) for i in range(5)], article('repeat')]
    excluded = [article('exclude'), article('alias', url='https://example.test/background0?utm_source=feed')]
    result = await news_map.build_news_map(center, pool=pool, target=4, exclude_articles=excluded,
                                           search_context={'query': '원유', 'next_positions': {'원유': 21}})
    assert len(result.items) == 4
    assert {item.article['news_id'] for item in result.items} == {f'background{i}' for i in range(1, 5)}
    assert not any(c['article_b']['title'] in ('exclude', 'background0') for c in calls)


@pytest.mark.asyncio
async def test_expansion_continues_map_query_instead_of_article_membership(monkeypatch):
    from app.agents import news_map
    from app.agents.related_selector import NewsMapSelector
    center = {'news_id': 'center', 'title': '유가', '_search_keyword': '다른 맵'}
    expansion = news_map._Expansion(center, NewsMapSelector(center), {},
                                    {'query': '원유', 'observed_at': state.iso(), 'next_positions': {'원유': 21}})
    recorded = []

    async def search(query, **kwargs):
        recorded.append((query, kwargs['start']))
        return []

    monkeypatch.setattr(expansion, 'search', search)
    steps = list(expansion.steps([], 0))
    await steps[0][2]()
    assert recorded == [('원유', 21)]


@pytest.mark.asyncio
async def test_map_search_context_and_cas_failure_do_not_cross_contaminate(fixture):
    client, _, _, _ = fixture
    first, second = create(client, '원유'), create(client, '금값')
    sid = client.cookies.get('econmind_sid')
    internal = await maps.load_map(sid, first['map_id'])
    assert internal['search_context']['query'] == '원유'
    assert (await maps.load_map(sid, second['map_id']))['search_context']['query'] == '금값'
    internal['detail_news_id'] = 'a1'
    await maps.commit_map(sid, internal, 1)
    with pytest.raises(Exception, match='변경'):
        await maps.commit_map(sid, internal, 1)
