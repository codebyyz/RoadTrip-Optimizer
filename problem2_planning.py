# -*- coding: utf-8 -*-
"""
问题二：无随机扰动下的多目标景点优选与基准行程设计
高效版本：days_content 无需枚举排列（指标与顺序无关）
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 数据加载 =====================
with open('code/scenic_spots.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

spots = data['spots']
drive_matrix = np.array(data['drive_matrix']['matrix'])
labels = data['drive_matrix']['labels']
idx_map = {lab: i for i, lab in enumerate(labels)}
spot_info = {s['id']: s for s in spots}
spot_labels = [s['id'] for s in spots]
n = len(spots)

DAY_START = 7.0
DAY_END = 21.0
MAX_DAY_HOURS = DAY_END - DAY_START
PREP_TIME = 1.5
MEAL_TIME = 1.0

def day_pattern_cost(spot_pair):
    if len(spot_pair) == 1:
        s = spot_pair[0]
        d = drive_matrix[idx_map['酒店'], idx_map[s]] * 2
        v = spot_info[s]['comfort_hours']
        total = PREP_TIME + d + v + 2*MEAL_TIME
        return d, total, (s,), spot_info[s]['preference']
    s1, s2 = spot_pair
    d1 = drive_matrix[idx_map['酒店'], idx_map[s1]] + drive_matrix[idx_map[s1], idx_map[s2]] + drive_matrix[idx_map[s2], idx_map['酒店']]
    d2 = drive_matrix[idx_map['酒店'], idx_map[s2]] + drive_matrix[idx_map[s2], idx_map[s1]] + drive_matrix[idx_map[s1], idx_map['酒店']]
    v = spot_info[s1]['comfort_hours'] + spot_info[s2]['comfort_hours']
    t1 = PREP_TIME + d1 + v + 2*MEAL_TIME
    t2 = PREP_TIME + d2 + v + 2*MEAL_TIME
    if t1 <= t2:
        return d1, t1, (s1, s2), spot_info[s1]['preference'] + spot_info[s2]['preference']
    else:
        return d2, t2, (s2, s1), spot_info[s1]['preference'] + spot_info[s2]['preference']

# ===================== 枚举分配方案 =====================
print("="*60)
print("【问题二】无随机扰动下的多目标景点优选与基准行程设计")
print("="*60)

all_solutions = []

for k in range(5, 9):
    m = k - 5
    for selected in combinations(spot_labels, k):
        sel = list(selected)
        # 生成恰好 m 个不相交 pair 的所有方式
        pair_lists = []
        if m == 0:
            pair_lists = [[]]
        elif m == 1:
            for i,a in enumerate(sel):
                for b in sel[i+1:]:
                    pair_lists.append([(a,b)])
        elif m == 2:
            for i,a in enumerate(sel):
                for j,b in enumerate(sel[i+1:], i+1):
                    rest1 = [sel[x] for x in range(len(sel)) if x!=i and x!=j]
                    for k1,c in enumerate(rest1):
                        for d in rest1[k1+1:]:
                            pair_lists.append([(a,b),(c,d)])
        elif m == 3:
            for i,a in enumerate(sel):
                for j,b in enumerate(sel[i+1:], i+1):
                    rest1 = [sel[x] for x in range(len(sel)) if x!=i and x!=j]
                    for k1,c in enumerate(rest1):
                        for l,d in enumerate(rest1[k1+1:], k1+1):
                            rest2 = [rest1[x] for x in range(len(rest1)) if x!=k1 and x!=l]
                            for n,e in enumerate(rest2):
                                for f in rest2[n+1:]:
                                    pair_lists.append([(a,b),(c,d),(e,f)])
        
        for pair_list in pair_lists:
            used = set()
            for a,b in pair_list:
                used.add(a); used.add(b)
            singles = [s for s in sel if s not in used]
            days_content = pair_list + [(s,) for s in singles]
            assert len(days_content) == 5
            
            total_pref = 0
            total_drive = 0
            daily_total = []
            daily_patterns = []
            valid = True
            for day_content in days_content:
                d, total, order, pref = day_pattern_cost(day_content)
                if total > MAX_DAY_HOURS:
                    valid = False
                    break
                total_pref += pref
                total_drive += d
                daily_total.append(total)
                daily_patterns.append({'drive':d, 'total':total, 'order':order, 'pref':pref})
            if not valid:
                continue
            avg_drive = total_drive / 5
            balance = -np.std(daily_total)
            all_solutions.append({
                'selected': sel,
                'patterns': daily_patterns,
                'total_pref': total_pref,
                'avg_drive': avg_drive,
                'balance': balance,
                'daily_drive': [p['drive'] for p in daily_patterns],
                'daily_total': daily_total
            })

print(f"总可行解数量: {len(all_solutions)}")

# ===================== 快速Pareto筛选 (三维) =====================
print("进行Pareto前沿筛选...")
import bisect
all_solutions.sort(key=lambda x: x['total_pref'], reverse=True)

pareto_list = []
for sol in all_solutions:
    d = sol['avg_drive']
    b = sol['balance']
    idx = bisect.bisect_right(pareto_list, (d, float('inf'), None))
    if idx > 0:
        max_b = max(item[1] for item in pareto_list[:idx])
        if max_b >= b:
            continue
    new_list = []
    for item in pareto_list:
        if item[0] >= d and item[1] <= b:
            continue
        new_list.append(item)
    new_list.append((d, b, sol))
    new_list.sort(key=lambda x: x[0])
    pareto_list = new_list

pareto = [item[2] for item in pareto_list]
print(f"Pareto 前沿解数量: {len(pareto)}")

# ===================== 综合评分选最优 =====================
pareto_df = pd.DataFrame(pareto)
pareto_df['neg_drive'] = -pareto_df['avg_drive']

for col in ['total_pref', 'neg_drive', 'balance']:
    mn = pareto_df[col].min()
    mx = pareto_df[col].max()
    if mx > mn:
        pareto_df[col + '_norm'] = (pareto_df[col] - mn) / (mx - mn)
    else:
        pareto_df[col + '_norm'] = 1.0

w1, w2, w3 = 0.5, 0.25, 0.25
pareto_df['score'] = w1 * pareto_df['total_pref_norm'] + w2 * pareto_df['neg_drive_norm'] + w3 * pareto_df['balance_norm']

best = pareto_df.loc[pareto_df['score'].idxmax()]
best_patterns = best['patterns']
selected_best = best['selected']

print("\n最优方案:")
print(f"  选中景点: {selected_best}")
print(f"  总喜好度: {best['total_pref']:.2f}")
print(f"  平均日车程: {best['avg_drive']:.2f}h")
print(f"  日时间标准差: {-best['balance']:.2f}h")
for i,p in enumerate(best_patterns):
    print(f"  Day{i+1}: {p['order']} 车程={p['drive']:.2f}h 总耗时={p['total']:.2f}h")

# ===================== 详细时序编排 =====================
print("\n" + "=" * 60)
print("【基准行程详细时序】")

schedule_result = []
for d in range(5):
    p = best_patterns[d]
    day_spots = list(p['order'])
    tl = []
    t = DAY_START
    tl.append((t, t + PREP_TIME, "起床+早餐+整装"))
    t += PREP_TIME

    d1 = drive_matrix[idx_map['酒店'], idx_map[day_spots[0]]]
    tl.append((t, t + d1, f"酒店→{day_spots[0]}"))
    t += d1

    v1 = spot_info[day_spots[0]]['comfort_hours']
    tl.append((t, t + v1, f"游览 {day_spots[0]}({spot_info[day_spots[0]]['name']})"))
    t += v1

    if len(day_spots) == 2:
        d2 = drive_matrix[idx_map[day_spots[0]], idx_map[day_spots[1]]]
        tl.append((t, t + d2, f"{day_spots[0]}→{day_spots[1]}"))
        t += d2
        tl.append((t, t + MEAL_TIME, "午餐/正餐"))
        t += MEAL_TIME
        v2 = spot_info[day_spots[1]]['comfort_hours']
        tl.append((t, t + v2, f"游览 {day_spots[1]}({spot_info[day_spots[1]]['name']})"))
        t += v2
        d3 = drive_matrix[idx_map[day_spots[1]], idx_map['酒店']]
        tl.append((t, t + d3, f"{day_spots[1]}→酒店"))
        t += d3
    else:
        tl.append((t, t + MEAL_TIME, "午餐/正餐"))
        t += MEAL_TIME
        d3 = drive_matrix[idx_map[day_spots[0]], idx_map['酒店']]
        tl.append((t, t + d3, f"{day_spots[0]}→酒店"))
        t += d3

    tl.append((t, t + MEAL_TIME, "晚餐"))
    t += MEAL_TIME
    tl.append((t, DAY_END, "自由休息/酒店"))

    schedule_result.append({
        'day': d + 1,
        'date': f'5月{d + 1}日',
        'spots': day_spots,
        'timeline': tl,
        'end_time': t
    })
    print(f"\n--- 5月{d + 1}日 (Day {d + 1}) 景点: {', '.join(day_spots)} ---")
    for start, end, desc in tl:
        print(f"  {start:5.2f}h - {end:5.2f}h | {desc}")

with open('code/problem2_baseline.json', 'w', encoding='utf-8') as f:
    json.dump({
        'selected': selected_best,
        'assignment': [list(p['order']) for p in best_patterns],
        'schedule': [
            {
                'day': s['day'], 'date': s['date'], 'spots': s['spots'],
                'timeline': [(float(a), float(b), c) for a, b, c in s['timeline']],
                'end_time': float(s['end_time'])
            } for s in schedule_result
        ]
    }, f, ensure_ascii=False, indent=2)

# ===================== 可视化 =====================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax0 = axes[0]
sc = ax0.scatter(pareto_df['total_pref'], pareto_df['avg_drive'],
                  c=pareto_df['balance'], cmap='RdYlGn', s=80, edgecolors='k')
ax0.scatter(best['total_pref'], best['avg_drive'], c='red', marker='*', s=300, edgecolors='k', label='最优解', zorder=5)
ax0.set_xlabel('总喜好满意度')
ax0.set_ylabel('平均日车程(h)')
ax0.set_title('Pareto前沿解分布（颜色=均衡性）')
ax0.legend()
ax0.grid(True, alpha=0.3)
cbar = plt.colorbar(sc, ax=ax0)
cbar.set_label('均衡性评分')

ax1 = axes[1]
colors_day = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
for d, s in enumerate(schedule_result):
    for start, end, desc in s['timeline']:
        ax1.barh(d, end - start, left=start, height=0.6, color=colors_day[d], alpha=0.7, edgecolor='k')
        if end - start > 0.5:
            ax1.text(start + (end - start) / 2, d, desc, ha='center', va='center', fontsize=7)
ax1.set_yticks(range(5))
ax1.set_yticklabels([f'Day {i + 1}' for i in range(5)])
ax1.set_xlabel('时间 (h)')
ax1.set_title('基准行程甘特图（理想无扰动）')
ax1.set_xlim(DAY_START, DAY_END)
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

plt.tight_layout()
plt.savefig('figture/problem2_baseline.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n已保存 figture/problem2_baseline.png")
print("=" * 60)
