from pathlib import Path

from pdmetrics.metrics.f1 import pdF1
from pdmetrics.syn.data import load_df

data_dir = Path('/tmp/classification')

db_path = Path("/tmp/classification/metrics.db")
f1 = pdF1.load_from_db(db_path)
dset = load_df(f1.db_path, "dataset")
print(f1)

rowids = [int(r) for r in dset["rowid"]]

scores = {}
for _, row in f1.df.iterrows():
    _row = dict(row.to_dict())
    _row.update({"rowid": row.name})
    scores[row.name] = _row

dataset = {}
for _, row in dset.iterrows():
    _row = dict(row.to_dict())
    _row.update({"rowid": row.name})
    dataset[row.name] = _row


from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(loader=PackageLoader("pdmetrics"), autoescape=select_autoescape())
template = env.get_template("index.html")
# print(template.render(scores=scores))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def root():
    return template.render(rowids=rowids, scores=scores, dataset=dataset)


@app.get("/{filename}/{table}", response_class=HTMLResponse)
async def selected(filename: str, table: str):
    sel_path = Path(f"/tmp/classification/{filename}.db")
    dset = load_df(sel_path, table)

    rows = {}
    for _, row in dset.iterrows():
        _row = dict(row.to_dict())
        _row.update({"rowid": row.name})
        rows[row.name] = _row
    rowids = list(rows.keys())

    template = env.get_template("rows.html")
    return template.render(rowids=rowids, rows=rows)
