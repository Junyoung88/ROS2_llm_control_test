import json
import shutil
from datetime import datetime
from collections import Counter

# 데이터 로드
backup = [json.loads(l) for l in open('experiment_results/gazebo_s1_s6/results_pre_s4v5_backup.jsonl')]
checkpoint = json.load(open('experiment_results/gazebo_s1_s6/checkpoint.json'))
new_done_ids = set(checkpoint['completed_trial_ids'])

# 새 실험 결과 (중복 시 새 것 우선)
lines = open('experiment_results/gazebo_s1_s6/results.jsonl').readlines()
new_results_by_id = {}
for l in lines:
    t = json.loads(l)
    if t['trial_id'] in new_done_ids:
        new_results_by_id[t['trial_id']] = t

# 백업에서 drift 없는 것 (S4 제외, drift 없는 것만 재사용)
backup_clean_by_id = {}
for t in backup:
    if t['scenario'] != 'S4':
        has_drift = 'drift' in str(t.get('reason', '')).lower()
        if not has_drift:
            # 새 실험에 없는 것만 백업 사용
            if t['trial_id'] not in new_results_by_id:
                backup_clean_by_id[t['trial_id']] = t

# 완료된 trial IDs = 새 실험 + 백업 clean
completed_ids = list(new_done_ids | set(backup_clean_by_id.keys()))
print(f'새 실험 완료: {len(new_done_ids)}')
print(f'백업 재사용: {len(backup_clean_by_id)}')
print(f'총 completed IDs: {len(completed_ids)}')
print(f'970 - {len(completed_ids)} = {970 - len(completed_ids)} 남은 trials')

# 결과 파일 병합 (새 실험 우선, 백업은 새 실험에 없는 것만)
merged = {}
for tid, t in backup_clean_by_id.items():
    merged[tid] = t
for tid, t in new_results_by_id.items():
    merged[tid] = t

print(f'병합 결과 파일: {len(merged)} unique trials')

# 기존 results.jsonl 백업
shutil.copy('experiment_results/gazebo_s1_s6/results.jsonl',
            'experiment_results/gazebo_s1_s6/results_pre_merge_backup.jsonl')
print('results.jsonl 백업 완료 → results_pre_merge_backup.jsonl')

# 새 results.jsonl 쓰기
with open('experiment_results/gazebo_s1_s6/results.jsonl', 'w') as f:
    for t in merged.values():
        f.write(json.dumps(t) + '\n')
print(f'results.jsonl 업데이트 완료 ({len(merged)} trials)')

# 체크포인트 업데이트
checkpoint['completed_trial_ids'] = completed_ids
checkpoint['completed_trials'] = len(completed_ids)
checkpoint['current_trial_idx'] = 0  # 전체 순회하되 completed 스킵
checkpoint['last_updated'] = datetime.now().isoformat()
with open('experiment_results/gazebo_s1_s6/checkpoint.json', 'w') as f:
    json.dump(checkpoint, f, indent=2)
print(f'checkpoint.json 업데이트 완료 (completed={len(completed_ids)}/970)')
