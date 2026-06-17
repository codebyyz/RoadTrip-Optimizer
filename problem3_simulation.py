# -*- coding: utf-8 -*-
"""
问题三：随机扰动下的行程稳定性量化评估
- 蒙特卡洛模拟道路堵车与景点排队
- 评估：游览时长不达标概率、当日超时概率、整体可靠度
- 量化堵车与排队的贡献度
- 识别结构性薄弱点
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# ===================== 数据加载 =====================
with open('code/scenic_spots.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

spots = data['spots']
drive_matrix = np.array(data['drive_matrix']['matrix'])
labels = data['drive_matrix']['labels']
idx_map = {lab:i for i,lab in enumerate(labels)}
spot_info = {s['id']: s for s in spots}

with open('code/problem2_baseline.json', 'r', encoding='utf-8') as f:
    baseline = json.load(f)

schedule = baseline['schedule']

# ===================== 随机规则 =====================
DAY_START = 7.0
DAY_END = 21.0
MAX_DAY_HOURS = DAY_END - DAY_START
PREP_TIME = 1.5
MEAL_TIME = 1.0

def is_peak_drive(t):
    """判断时间是否处于道路高峰"""
    # 高峰: 7-9, 11-13, 16-18
    return (7 <= t < 9) or (11 <= t < 13) or (16 <= t < 18)

def is_peak_entry(t):
    """判断入园时间是否高峰"""
    return 9 <= t < 12

def sample_drive_delay(t_start, duration):
    """采样某段车程的堵车延时"""
    # 简化：取车程中点判断时段
    mid = t_start + duration/2
    if is_peak_drive(mid):
        # 高峰堵车额外延时 1~4h 均匀分布（题目说"发生拥堵时"，这里简化为一定发生或有概率）
        # 五一实际高发，设堵车概率 0.6
        if np.random.rand() < 0.6:
            return np.random.uniform(1.0, 4.0)
        else:
            return 0.0
    else:
        if np.random.rand() < 0.3:
            return np.random.uniform(0.0, 1.5)
        else:
            return 0.0

def sample_queue_delay(t_entry):
    """采样入园排队延时"""
    if is_peak_entry(t_entry):
        # 高峰入园排队 0.5~3h
        return np.random.uniform(0.5, 3.0)
    else:
        # 平峰 0~1h
        return np.random.uniform(0.0, 1.0)

# ===================== 单次模拟 =====================
def simulate_one():
    """对基准行程做一次随机模拟，返回每日结果"""
    day_results = []
    for day_sched in schedule:
        day = day_sched['day']
        spots_day = day_sched['spots']
        t = DAY_START + PREP_TIME  # 出发时刻
        
        total_time = PREP_TIME
        visit_shortfall = 0  # 游览时长缺口
        timeout = False
        
        # 去第一个景点
        d1 = drive_matrix[idx_map['酒店'], idx_map[spots_day[0]]]
        delay1 = sample_drive_delay(t, d1)
        total_time += d1 + delay1
        t += d1 + delay1
        
        # 入园排队+游览景点1
        q1 = sample_queue_delay(t)
        total_time += q1
        t += q1
        visit1_req = spot_info[spots_day[0]]['min_hours']
        visit1_comf = spot_info[spots_day[0]]['comfort_hours']
        # 实际可用游览时间受后续行程挤压，这里简化：如果总时间未超，则给舒适时长；否则压缩到最低
        # 更准确的：模拟按顺序走，到最后看是否超时
        total_time += visit1_comf
        t += visit1_comf
        
        if len(spots_day) == 2:
            # 景点间通勤
            d2 = drive_matrix[idx_map[spots_day[0]], idx_map[spots_day[1]]]
            delay2 = sample_drive_delay(t, d2)
            total_time += d2 + delay2 + MEAL_TIME
            t += d2 + delay2 + MEAL_TIME
            
            # 景点2排队+游览
            q2 = sample_queue_delay(t)
            total_time += q2
            t += q2
            visit2_req = spot_info[spots_day[1]]['min_hours']
            visit2_comf = spot_info[spots_day[1]]['comfort_hours']
            total_time += visit2_comf
            t += visit2_comf
            
            # 返程
            d3 = drive_matrix[idx_map[spots_day[1]], idx_map['酒店']]
            delay3 = sample_drive_delay(t, d3)
            total_time += d3 + delay3
            t += d3 + delay3
        else:
            # 午餐+返程
            total_time += MEAL_TIME
            t += MEAL_TIME
            d3 = drive_matrix[idx_map[spots_day[0]], idx_map['酒店']]
            delay3 = sample_drive_delay(t, d3)
            total_time += d3 + delay3
            t += d3 + delay3
        
        # 晚餐
        total_time += MEAL_TIME
        t += MEAL_TIME
        
        # 判定
        # 游览时长不达标：由于我们固定给了舒适时长，这里改为检查"在超时压力下是否被迫压缩"
        # 更合理的判定：如果 total_time > MAX_DAY_HOURS，则视为需要压缩游览时间
        if total_time > MAX_DAY_HOURS:
            timeout = True
            excess = total_time - MAX_DAY_HOURS
            # 被迫压缩游览时间（优先压缩舒适部分）
            # 当天总舒适游览时长
            total_comf = sum(spot_info[s]['comfort_hours'] for s in spots_day)
            total_min = sum(spot_info[s]['min_hours'] for s in spots_day)
            if total_comf - excess < total_min:
                visit_shortfall = total_min - (total_comf - excess)
                # 即：即使压到最低，仍然不够的时长
        
        day_results.append({
            'day': day,
            'total_time': total_time,
            'timeout': timeout,
            'visit_shortfall': max(0, visit_shortfall)
        })
    return day_results

# ===================== 蒙特卡洛模拟 =====================
print("="*60)
print("【问题三】随机扰动下的行程稳定性量化评估")
print("="*60)

N_SIM = 20000
all_results = []
for _ in range(N_SIM):
    all_results.append(simulate_one())

# 统计
day_timeout_probs = []
day_shortfall_probs = []
overall_reliability = []

for day in range(5):
    timeouts = [r[day]['timeout'] for r in all_results]
    shortfalls = [r[day]['visit_shortfall'] > 0 for r in all_results]
    day_timeout_probs.append(np.mean(timeouts))
    day_shortfall_probs.append(np.mean(shortfalls))

# 整体可靠度：5天都不超时且游览时长达标
reliable_count = 0
for r in all_results:
    ok = all(not d['timeout'] and d['visit_shortfall'] == 0 for d in r)
    if ok:
        reliable_count += 1
overall_rel = reliable_count / N_SIM

print(f"\n蒙特卡洛模拟次数: {N_SIM}")
print(f"\n各日超时概率:")
for d in range(5):
    print(f"  Day {d+1}: {day_timeout_probs[d]*100:.2f}%")
print(f"\n各日游览时长不达标概率:")
for d in range(5):
    print(f"  Day {d+1}: {day_shortfall_probs[d]*100:.2f}%")
print(f"\n行程整体可靠度（5天全部正常）: {overall_rel*100:.2f}%")

# ===================== 贡献度分解 =====================
print("\n" + "="*60)
print("【扰动因素贡献度分解】")

def simulate_one_factor(factor='both'):
    """factor: 'drive_only', 'queue_only', 'both', 'none'"""
    day_results = []
    for day_sched in schedule:
        spots_day = day_sched['spots']
        t = DAY_START + PREP_TIME
        total_time = PREP_TIME
        
        d1 = drive_matrix[idx_map['酒店'], idx_map[spots_day[0]]]
        delay1 = sample_drive_delay(t, d1) if factor in ('drive_only','both') else 0.0
        if factor == 'queue_only':
            delay1 = 0.0
        total_time += d1 + delay1
        t += d1 + delay1
        
        q1 = sample_queue_delay(t) if factor in ('queue_only','both') else 0.0
        if factor == 'drive_only':
            q1 = 0.0
        total_time += q1 + spot_info[spots_day[0]]['comfort_hours']
        t += q1 + spot_info[spots_day[0]]['comfort_hours']
        
        if len(spots_day)==2:
            d2 = drive_matrix[idx_map[spots_day[0]], idx_map[spots_day[1]]]
            delay2 = sample_drive_delay(t, d2) if factor in ('drive_only','both') else 0.0
            if factor == 'queue_only': delay2=0.0
            total_time += d2 + delay2 + MEAL_TIME
            t += d2 + delay2 + MEAL_TIME
            
            q2 = sample_queue_delay(t) if factor in ('queue_only','both') else 0.0
            if factor == 'drive_only': q2=0.0
            total_time += q2 + spot_info[spots_day[1]]['comfort_hours']
            t += q2 + spot_info[spots_day[1]]['comfort_hours']
            
            d3 = drive_matrix[idx_map[spots_day[1]], idx_map['酒店']]
            delay3 = sample_drive_delay(t, d3) if factor in ('drive_only','both') else 0.0
            if factor == 'queue_only': delay3=0.0
            total_time += d3 + delay3
            t += d3 + delay3
        else:
            total_time += MEAL_TIME
            t += MEAL_TIME
            d3 = drive_matrix[idx_map[spots_day[0]], idx_map['酒店']]
            delay3 = sample_drive_delay(t, d3) if factor in ('drive_only','both') else 0.0
            if factor == 'queue_only': delay3=0.0
            total_time += d3 + delay3
            t += d3 + delay3
        
        total_time += MEAL_TIME
        timeout = total_time > MAX_DAY_HOURS
        day_results.append(timeout)
    return any(day_results)  # 只要有一天超时即认为行程受影响

# 分别模拟
N = 20000
results_drive = [simulate_one_factor('drive_only') for _ in range(N)]
results_queue = [simulate_one_factor('queue_only') for _ in range(N)]
results_both = [simulate_one_factor('both') for _ in range(N)]
results_none = [simulate_one_factor('none') for _ in range(N)]

P_drive = np.mean(results_drive)
P_queue = np.mean(results_queue)
P_both = np.mean(results_both)
P_none = np.mean(results_none)

print(f"无扰动超时概率: {P_none*100:.2f}%")
print(f"仅道路堵车影响概率: {P_drive*100:.2f}%")
print(f"仅景点排队影响概率: {P_queue*100:.2f}%")
print(f"两者共同影响概率: {P_both*100:.2f}%")

# Shapley-like 贡献度
# 总扰动 = P_both - P_none
# 堵车单独贡献 = P_drive - P_none
# 排队单独贡献 = P_queue - P_none
# 交互贡献 = (P_both - P_none) - (P_drive - P_none) - (P_queue - P_none)
contrib_drive = P_drive - P_none
contrib_queue = P_queue - P_none
contrib_interact = (P_both - P_none) - contrib_drive - contrib_queue
print(f"\n扰动贡献度分解:")
print(f"  道路堵车独立贡献: {contrib_drive*100:.2f}%")
print(f"  景点排队独立贡献: {contrib_queue*100:.2f}%")
print(f"  两者交互贡献: {contrib_interact*100:.2f}%")

# ===================== 可视化 =====================
fig, axes = plt.subplots(1, 3, figsize=(18,5))

# 左：各日风险
ax0 = axes[0]
x = np.arange(1,6)
width = 0.35
bars1 = ax0.bar(x - width/2, [p*100 for p in day_timeout_probs], width, label='超时概率', color='#e74c3c')
bars2 = ax0.bar(x + width/2, [p*100 for p in day_shortfall_probs], width, label='游览不达标概率', color='#f39c12')
ax0.set_xlabel('日期')
ax0.set_ylabel('概率 (%)')
ax0.set_title('各日风险概率分布')
ax0.set_xticks(x)
ax0.legend()
ax0.grid(True, alpha=0.3, axis='y')
# 标注数值
for bar in bars1:
    h = bar.get_height()
    ax0.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax0.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

# 中：可靠度饼图
ax1 = axes[1]
labels_pie = ['可靠', '不可靠']
sizes = [overall_rel*100, (1-overall_rel)*100]
colors_pie = ['#2ecc71', '#e74c3c']
explode = (0.05, 0)
ax1.pie(sizes, explode=explode, labels=labels_pie, colors=colors_pie, autopct='%1.2f%%', shadow=True, startangle=90)
ax1.set_title(f'行程整体可靠度\n(目标≥90%)')

# 右：贡献度堆叠条形
ax2 = axes[2]
categories = ['道路堵车', '景点排队', '交互效应']
values = [contrib_drive*100, contrib_queue*100, contrib_interact*100]
colors_bar = ['#3498db', '#9b59b6', '#e67e22']
ax2.barh(categories, values, color=colors_bar, edgecolor='k')
ax2.set_xlabel('贡献度 (%)')
ax2.set_title('扰动因素贡献度分解')
ax2.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(values):
    ax2.text(v + 0.2, i, f'{v:.2f}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figture/problem3_simulation.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n已保存 figture/problem3_simulation.png")

# ===================== 结构性薄弱点识别 =====================
print("\n" + "="*60)
print("【结构性薄弱点识别】")
# 找出超时概率最高的天数
day_risk = [(i+1, day_timeout_probs[i]) for i in range(5)]
day_risk.sort(key=lambda x: x[1], reverse=True)
print("各日超时风险排名:")
for d, p in day_risk:
    print(f"  Day {d}: {p*100:.2f}%")
print(f"\n最薄弱的环节为 Day {day_risk[0][0]}，超时概率最高，建议:")
print("  1. 减少该日景点数量或更换为近距离景点；")
print("  2. 提前出发避开高峰；")
print("  3. 预留弹性时间缓冲。")

# 保存数据
pd.DataFrame({
    'Day': range(1,6),
    '超时概率': [p*100 for p in day_timeout_probs],
    '游览不达标概率': [p*100 for p in day_shortfall_probs]
}).to_csv('code/problem3_risk.csv', index=False, encoding='utf-8-sig')

print("="*60)
