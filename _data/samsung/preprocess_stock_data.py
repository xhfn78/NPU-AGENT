import csv
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).parent
START_DATE = datetime.strptime("2024-10-12", "%Y-%m-%d").date()
END_DATE = datetime.strptime("2026-10-12", "%Y-%m-%d").date()
FEATURE_COLUMNS = ["시가", "고가", "저가", "종가", "거래량"]


def read_stock(filename: str, prefix: str) -> dict:
    """Read one KRX-export CSV, clean numeric text, and index it by date."""
    data = {}
    with (BASE_DIR / filename).open(encoding="cp949", newline="") as source:
        for row in csv.DictReader(source):
            try:
                date = datetime.strptime(row["일자"], "%Y/%m/%d").date()
                values = {f"{prefix}_{column}": int(row[column].replace(",", "")) for column in FEATURE_COLUMNS}
            except (KeyError, TypeError, ValueError):
                continue
            if START_DATE <= date <= END_DATE:
                data[date] = values
    return data


amore = read_stock("아모레261012.csv", "아모레")
hynix = read_stock("하이닉스261012.csv", "하이닉스")
samsung = read_stock("삼성전자261012.csv", "삼성전자")

# On each trade date, use only information known that day.
# The label is Samsung's opening price on its next trading date.
dates = sorted(samsung)
next_samsung = {date: dates[index + 1] for index, date in enumerate(dates[:-1])}
dataset = []
for date in sorted(set(amore) & set(hynix) & set(next_samsung)):
    next_date = next_samsung[date]
    dataset.append({
        "일자": date.isoformat(),
        **amore[date],
        **hynix[date],
        "다음_삼성전자_일자": next_date.isoformat(),
        "다음_삼성전자_시가": samsung[next_date]["삼성전자_시가"],
    })

output_path = BASE_DIR / "삼성전자_다음시가_전처리_20241012_20261012.csv"
fieldnames = ["일자", *[f"아모레_{column}" for column in FEATURE_COLUMNS], *[f"하이닉스_{column}" for column in FEATURE_COLUMNS], "다음_삼성전자_일자", "다음_삼성전자_시가"]
with output_path.open("w", encoding="utf-8-sig", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(dataset)

print(f"amore rows: {len(amore):,}")
print(f"hynix rows: {len(hynix):,}")
print(f"samsung rows: {len(samsung):,}")
print(f"model rows: {len(dataset):,}")
print(f"feature dates: {dataset[0]['일자']} to {dataset[-1]['일자']}")
print(f"target dates: {dataset[0]['다음_삼성전자_일자']} to {dataset[-1]['다음_삼성전자_일자']}")
for row in dataset[:3]:
    print(row)
