import git
import csv
import numpy as np
from cottools.scc import collect_scc, SccData
from cottools.filesolver import NameRepo
from concernbert.frontend import CdCalculator
from scipy.stats import pearsonr

from datetime import datetime
from collections import defaultdict

REPO = "_repos/nifi"

repo = git.Repo(REPO, odbt=git.GitCmdObjectDB)
scc_data = collect_scc(REPO)
file_repo = NameRepo.parse_log(REPO, "HEAD")
cd_calculator = CdCalculator("_models/EntityBERT-v3_train_nonldl-lr5e5-2_83-e3", "_cache")

def load_bytes(repo: git.Repo, hexsha: str) -> bytes:
    return repo.odb.stream(bytes.fromhex(hexsha)).read()  # type: ignore

results = []

for file_id in file_repo.all_file_ids():
    try:
        file_name = file_repo.file_name_by_id(file_repo.latest_commit(), file_id)
        cont_changes = file_repo.cont_changes_by_id(file_id)

        metrics = defaultdict(list)
        for rev in cont_changes:
            try:
                commit = repo.commit(rev)
                path = file_repo.file_name_by_id(rev, file_id)
                obj = commit.tree.join(path)
                if not isinstance(obj, git.Blob):
                    continue

                content = load_bytes(repo, obj.hexsha)
                source = content.decode("utf-8", errors="ignore")
                cd = cd_calculator.calc_cd(source, pbar=False)

                metrics['loc'].append(scc_data[obj.hexsha].loc)
                metrics['lloc'].append(scc_data[obj.hexsha].loc)  # reuse if lloc not separated
                metrics['entities'].append(cd.num_entities)
                metrics['intra_cd'].append(cd.intra_cd)
                metrics['inter_cd'].append(cd.inter_cd)
            except Exception:
                continue

        if len(metrics['loc']) < 2:
            continue  # not enough data points

        def safe_corr(x, y):
            try:
                return round(pearsonr(x, y)[0], 3)
            except:
                return ''

        results.append([
            file_name,
            len(metrics['loc']),
            safe_corr(metrics['loc'], metrics['intra_cd']),
            safe_corr(metrics['loc'], metrics['inter_cd']),
            safe_corr(metrics['lloc'], metrics['intra_cd']),
            safe_corr(metrics['lloc'], metrics['inter_cd']),
            safe_corr(metrics['entities'], metrics['intra_cd']),
            safe_corr(metrics['entities'], metrics['inter_cd']),
        ])
    except Exception:
        continue

with open("correlation_metrics_over_time.csv", mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow([
        'File', '# of Commits',
        'corr(LOC, IntraCD)', 'corr(LOC, InterCD)',
        'corr(LLOC, IntraCD)', 'corr(LLOC, InterCD)',
        'corr(Entities, IntraCD)', 'corr(Entities, InterCD)'
    ])
    writer.writerows(results)

print("CSV written to correlation_metrics_over_time.csv")
