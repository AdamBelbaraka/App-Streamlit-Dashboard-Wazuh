import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


DEFAULT_INDEX_PATTERN = "wazuh-alerts-4.x-*"


class WazuhIndexerClient:
    def __init__(
        self,
        base_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify_tls: bool = False,
        timeout: int = 20,
        index_pattern: str = DEFAULT_INDEX_PATTERN,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.verify_tls = verify_tls
        self.timeout = timeout
        self.index_pattern = index_pattern
        self.session = requests.Session()
        if username and password:
            self.session.auth = (username, password)

    def _request(self, method: str, path: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(
                method,
                url,
                json=json,
                timeout=self.timeout,
                verify=self.verify_tls,
            )
            if response.status_code == 401:
                raise PermissionError("Unauthorized: verify Wazuh Indexer credentials")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise ConnectionError(f"Indexer request failed: {exc}") from exc

    def test_connection(self) -> Dict[str, Any]:
        return self._request("GET", "/")

    @staticmethod
    def _build_filters(
        start: datetime,
        end: datetime,
        agents: Optional[List[str]] = None,
        mitre_ids: Optional[List[str]] = None,
        rule_ids: Optional[List[str]] = None,
        levels: Optional[List[int]] = None,
        atomic_only: bool = False,
    ) -> List[Dict[str, Any]]:
        filters: List[Dict[str, Any]] = [
            {
                "range": {
                    "@timestamp": {
                        "gte": start.isoformat(),
                        "lte": end.isoformat(),
                    }
                }
            }
        ]

        if atomic_only:
            filters.append({"exists": {"field": "rule.mitre.id"}})

        if agents:
            filters.append({"terms": {"agent.name": agents}})
        if mitre_ids:
            filters.append({"terms": {"rule.mitre.id": mitre_ids}})
        if rule_ids:
            filters.append({"terms": {"rule.id": rule_ids}})
        if levels:
            filters.append({"terms": {"rule.level": levels}})
        return filters

    def _search_page(
        self,
        start: datetime,
        end: datetime,
        agents: Optional[List[str]] = None,
        mitre_ids: Optional[List[str]] = None,
        rule_ids: Optional[List[str]] = None,
        levels: Optional[List[int]] = None,
        atomic_only: bool = False,
        size: int = 1000,
        search_after: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        filters = self._build_filters(start, end, agents, mitre_ids, rule_ids, levels, atomic_only)
        query: Dict[str, Any] = {
            "size": size,
            "query": {"bool": {"filter": filters}},
            "sort": [
                {"@timestamp": "asc"},
                {"_id": "asc"},
            ],
            "_source": True,
        }
        if search_after:
            query["search_after"] = search_after

        path = f"/{self.index_pattern}/_search"
        return self._request("POST", path, json=query)

    def fetch_alerts(
        self,
        start: datetime,
        end: datetime,
        agents: Optional[List[str]] = None,
        mitre_ids: Optional[List[str]] = None,
        rule_ids: Optional[List[str]] = None,
        levels: Optional[List[int]] = None,
        atomic_only: bool = False,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        next_search_after: Optional[List[Any]] = None

        while len(hits) < limit:
            response = self._search_page(
                start,
                end,
                agents=agents,
                mitre_ids=mitre_ids,
                rule_ids=rule_ids,
                levels=levels,
                atomic_only=atomic_only,
                size=min(1000, limit - len(hits)),
                search_after=next_search_after,
            )
            page_hits = response.get("hits", {}).get("hits", [])
            if not page_hits:
                break
            hits.extend(page_hits)
            next_search_after = page_hits[-1].get("sort")
            if not next_search_after:
                break
        return hits

    def fetch_all_paginated(
        self,
        start: datetime,
        end: datetime,
        agents: Optional[List[str]] = None,
        mitre_ids: Optional[List[str]] = None,
        rule_ids: Optional[List[str]] = None,
        levels: Optional[List[int]] = None,
        atomic_only: bool = False,
        page_size: int = 2000,
    ) -> Iterable[Tuple[List[Dict[str, Any]], Optional[List[Any]]]]:
        next_search_after: Optional[List[Any]] = None
        while True:
            response = self._search_page(
                start,
                end,
                agents=agents,
                mitre_ids=mitre_ids,
                rule_ids=rule_ids,
                levels=levels,
                atomic_only=atomic_only,
                size=page_size,
                search_after=next_search_after,
            )
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                break
            next_search_after = hits[-1].get("sort")
            yield hits, next_search_after
            if not next_search_after:
                break

    @staticmethod
    def flatten_alert(hit: Dict[str, Any]) -> Dict[str, Any]:
        source = hit.get("_source", {})
        agent = source.get("agent", {}) or {}
        rule = source.get("rule", {}) or {}
        mitre = (rule.get("mitre") or {}).get("id") if isinstance(rule.get("mitre"), dict) else None
        mitre_ids: List[str] = []
        if isinstance(mitre, list):
            mitre_ids = [str(item) for item in mitre]
        elif isinstance(mitre, str):
            mitre_ids = [mitre]

        win_data = ((source.get("data") or {}).get("win") or {}).get("eventdata") or {}

        return {
            "@timestamp": source.get("@timestamp"),
            "agent.name": agent.get("name"),
            "agent.id": agent.get("id"),
            "rule.id": rule.get("id"),
            "rule.description": rule.get("description"),
            "rule.level": rule.get("level"),
            "rule.mitre.id": mitre_ids,
            "location": source.get("location"),
            "data.win.eventdata.commandLine": win_data.get("commandLine"),
            "raw": source,
        }
