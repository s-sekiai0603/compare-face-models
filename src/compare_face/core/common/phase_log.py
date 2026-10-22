"""レポート出力の各フェーズの開始/終了時刻の記録（Excel「ログ」シート用）。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PhaseRecord:
    name: str
    start: datetime
    end: datetime | None = None
    note: str = ""

    @property
    def elapsed_sec(self) -> float | None:
        return (self.end - self.start).total_seconds() if self.end else None


class PhaseLogger:
    def __init__(self, echo=print):
        self.records: list[PhaseRecord] = []
        self.started_at = datetime.now()
        self.finished_at: datetime | None = None
        self.echo = echo

    @contextmanager
    def phase(self, name: str):
        rec = PhaseRecord(name=name, start=datetime.now())
        self.records.append(rec)
        self.echo(f"[{rec.start:%H:%M:%S}] 開始: {name}")
        try:
            yield rec
        except BaseException as e:
            rec.note = (rec.note + " " if rec.note else "") + f"失敗: {type(e).__name__}: {e}"
            raise
        finally:
            rec.end = datetime.now()
            self.echo(f"[{rec.end:%H:%M:%S}] 終了: {name} ({rec.elapsed_sec:.2f}秒)")

    def finish(self) -> None:
        self.finished_at = datetime.now()
