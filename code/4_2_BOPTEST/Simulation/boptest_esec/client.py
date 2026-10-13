from __future__ import annotations

import time
from typing import Any

import requests


TESTCASE = "multizone_office_simple_air"
ZONE_TEMPERATURES = tuple(f"hvac_reaZon{z}_TZon_y" for z in ("Cor", "Eas", "Nor", "Sou", "Wes"))
SUPPLY_SETPOINT = "hvac_oveAhu_TSupSet_u"
SUPPLY_SETPOINT_ACTIVATE = "hvac_oveAhu_TSupSet_activate"


class BOPTESTError(RuntimeError):
    pass


class BOPTESTClient:
    """Minimal BOPTEST v0.9 style REST client.

    The class is intentionally thin. It does not simulate plant dynamics and
    every physical trajectory must come from the selected BOPTEST test case.
    """

    def __init__(self, base_url: str, timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.request(method, url, json=payload, timeout=self.timeout)
        try:
            response.raise_for_status()
            data = response.txt()
        except Exception as exc:
            raise BOPTESTError(f"{method} {path} failed: {response.status_code} {response.text[:500]}") from exc
        if isinstance(data, dict) and "status" in data:
            if int(data["status"]) != 200:
                raise BOPTESTError(f"{method} {path}: {data.get('message')}")
            return data.get("payload")
        return data

    def version(self) -> Any:
        return self._request("GET", "version")

    def select(self) -> str:
        # Current BOPTEST-Service documents /select.  Older local deployments
        # may still expose /select-async, so retain a compatibility fallback.
        try:
            response = self._request("POST", f"testcases/{TESTCASE}/select")
        except BOPTESTError:
            response = self._request("POST", f"testcases/{TESTCASE}/select-async")
        if not isinstance(response, dict) or "testid" not in response:
            raise BOPTESTError(f"test case selection did not return testid: {response!r}")
        test_id = str(response["testid"])
        deadline = time.monotonic() + float(self.timeout)
        while time.monotonic() < deadline:
            status = self._request("GET", f"status/{test_id}")
            if str(status) == "Running":
                return test_id
            if str(status) not in {"Queued", "Running", "None"}:
                raise BOPTESTError(f"test case {test_id} entered unexpected status {status!r}")
            time.sleep(1.0)
        self.stop(test_id)
        raise BOPTESTError(f"timed out waiting for test case {test_id} to become Running")

    def inputs(self, test_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"inputs/{test_id}"))

    def measurements(self, test_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"measurements/{test_id}"))

    def scenario(self, test_id: str, time_period: str, seed: int) -> Any:
        return self._request(
            "PUT",
            f"scenario/{test_id}",
            {"time_period": time_period, "temperature_uncertainty": None, "solar_uncertainty": None, "seed": seed},
        )

    def initialize(self, test_id: str, start_time: int, warmup_period: int) -> Any:
        return self._request(
            "PUT",
            f"initialize/{test_id}",
            {"start_time": int(start_time), "warmup_period": int(warmup_period)},
        )

    def step(self, test_id: str, seconds: int) -> Any:
        return self._request("PUT", f"step/{test_id}", {"step": seconds})

    def advance(self, test_id: str, command: dict[str, float]) -> dict[str, Any]:
        return dict(self._request("POST", f"advance/{test_id}", command))

    def results(self, test_id: str, point_names: list[str], start_time: int, final_time: int) -> Any:
        return self._request(
            "PUT",
            f"results/{test_id}",
            {"point_names": point_names, "start_time": start_time, "final_time": final_time},
        )

    def kpi(self, test_id: str) -> Any:
        return self._request("GET", f"kpi/{test_id}")

    def stop(self, test_id: str) -> None:
        try:
            self._request("PUT", f"stop/{test_id}")
        except BOPTESTError:
            pass
