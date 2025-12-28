import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class WazuhManagerClient:
    def __init__(
        self,
        base_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify_tls: bool = False,
        timeout: int = 15,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.verify_tls = verify_tls
        self.timeout = timeout
        self._token: Optional[str] = None

    def authenticate(self) -> str:
        if not self.username or not self.password:
            raise ValueError("Username and password are required for authentication")

        url = f"{self.base_url}/security/user/authenticate"
        try:
            response = requests.post(
                url,
                auth=(self.username, self.password),
                timeout=self.timeout,
                verify=self.verify_tls,
            )
            if response.status_code == 401:
                raise PermissionError("Unauthorized: check Wazuh Manager credentials")
            response.raise_for_status()
            token = response.json().get("data", {}).get("token")
            if not token:
                raise RuntimeError("Failed to obtain authentication token from Wazuh Manager")
            self._token = token
            return token
        except requests.RequestException as exc:
            raise ConnectionError(f"Authentication request failed: {exc}") from exc

    def _authorized_headers(self) -> Dict[str, str]:
        if not self._token:
            self.authenticate()
        assert self._token
        return {"Authorization": f"Bearer {self._token}"}

    def test_health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/?pretty=true"
        headers = self._authorized_headers()
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_tls,
            )
            if response.status_code == 401:
                # Token may be expired; retry once after re-authentication
                self.authenticate()
                headers = self._authorized_headers()
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=self.timeout,
                    verify=self.verify_tls,
                )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise ConnectionError(f"Health check failed: {exc}") from exc

    def list_agents(self, limit: int = 50) -> Dict[str, Any]:
        url = f"{self.base_url}/agents?limit={limit}&offset=0"
        headers = self._authorized_headers()
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_tls,
            )
            if response.status_code == 401:
                self.authenticate()
                headers = self._authorized_headers()
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=self.timeout,
                    verify=self.verify_tls,
                )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise ConnectionError(f"Unable to list agents: {exc}") from exc
