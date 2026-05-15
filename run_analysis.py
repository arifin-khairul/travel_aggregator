import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 13
sns.set_style('whitegrid')

bookings = pd.read_csv('Dataset/Datasets/Bookings.csv')
sessions = pd.read_csv('Dataset/Datasets/Sessions.csv')

bookings['booking_time'] = pd.to_datetime(bookings['booking_time'], format='ISO8601', utc=True)
sessions['search_time'] = pd.to_datetime(sessions['search_time'], format='ISO8601', utc=True)
sessions['session_starting_time'] = pd.to_datetime(sessions['session_starting_time'], format='ISO8601', utc=True)

print('Bookings shape:', bookings.shape)
print('Sessions shape:', sessions.shape)
print()

# Q1
distinct_bookings = bookings['booking_id'].nunique()
distinct_sessions = sessions['session_id'].nunique()
distinct_searches = sessions['search_id'].nunique()
print('Q1 - Distinct counts')
print(f'  Bookings : {distinct_bookings}')
print(f'  Sessions : {distinct_sessions}')
print(f'  Searches : {distinct_searches}')
print()

# Q2
bookings_per_session = (
    sessions.dropna(subset=['booking_id'])
    .groupby('session_id')['booking_id']
    .nunique()
)
sessions_multi = (bookings_per_session > 1).sum()
print('Q2 - Sessions with more than one booking:', sessions_multi)
print('  Breakdown:', dict(bookings_per_session.value_counts().sort_index()))
print()

# Q3
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
bookings['day_of_week'] = bookings['booking_time'].dt.day_name()
bookings_by_day = bookings['day_of_week'].value_counts().reindex(day_order)
print('Q3 - Bookings by day of week:')
for day, cnt in bookings_by_day.items():
    print(f'  {day:12s}: {cnt}')
print(f'  Highest: {bookings_by_day.idxmax()} ({bookings_by_day.max()} bookings)')

colors = plt.cm.tab10.colors[:7]
fig, ax = plt.subplots(figsize=(9, 9))
ax.pie(
    bookings_by_day,
    labels=bookings_by_day.index,
    autopct='%1.1f%%',
    colors=colors,
    explode=[0.05] * 7,
    startangle=140,
    textprops={'fontsize': 11}
)
ax.set_title('Booking Distribution by Day of Week', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('q3_booking_by_day_pie.png', dpi=150)
plt.close()
print('  Saved: q3_booking_by_day_pie.png')
print()

# Q4
service_summary = (
    bookings.groupby('service_name')
    .agg(Total_Bookings=('booking_id', 'count'), Total_GBV_INR=('INR_Amount', 'sum'))
    .reset_index()
    .sort_values('Total_Bookings', ascending=False)
)
service_summary['Total_GBV_INR'] = service_summary['Total_GBV_INR'].round(2)
print('Q4 - Bookings and GBV per service:')
print(service_summary.to_string(index=False))
print()

# Q5
multi_booking_customers = (
    bookings.groupby('customer_id')['booking_id']
    .count()
    .loc[lambda x: x > 1]
    .index
)
multi_bookings = bookings[bookings['customer_id'].isin(multi_booking_customers)].copy()
multi_bookings['route'] = multi_bookings['from_city'] + ' -> ' + multi_bookings['to_city']
top_route = multi_bookings['route'].value_counts()
print('Q5 - Most booked route (repeat customers):')
print(f'  {top_route.idxmax()} ({top_route.max()} bookings)')
print(top_route.head(10).to_string())
print()

# Q6
city_advance = (
    bookings.groupby('from_city')['days_to_departure']
    .agg(avg_days_to_departure='mean', num_departures='count')
    .reset_index()
)
top3 = (
    city_advance[city_advance['num_departures'] >= 5]
    .sort_values('avg_days_to_departure', ascending=False)
    .head(3)
    .reset_index(drop=True)
)
top3.index += 1
top3['avg_days_to_departure'] = top3['avg_days_to_departure'].round(2)
print('Q6 - Top 3 advance-booking cities:')
print(top3.to_string())
print()

# Q7
num_cols = bookings.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = bookings[num_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, linewidths=0.5)
ax.set_title('Correlation Heatmap - Numerical Columns', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('q7_correlation_heatmap.png', dpi=150)
plt.close()

corr_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
max_pair = corr_upper.stack().idxmax()
print(f'Q7 - Max correlation: {max_pair[0]} & {max_pair[1]} (r = {corr_upper.stack().max():.4f})')
print('  Saved: q7_correlation_heatmap.png')
print()

# Q8
device_by_service = (
    bookings.groupby(['service_name', 'device_type_used'])['booking_id']
    .count()
    .reset_index(name='booking_count')
)
most_used_device = (
    device_by_service
    .sort_values('booking_count', ascending=False)
    .groupby('service_name')
    .first()
    .reset_index()
    .rename(columns={'device_type_used': 'most_used_device', 'booking_count': 'bookings'})
)
print('Q8 - Most used device per service:')
print(most_used_device.to_string(index=False))
print()

# Q9
bookings['year_quarter'] = bookings['booking_time'].dt.to_period('Q')
quarterly_device = (
    bookings.groupby(['year_quarter', 'device_type_used'])['booking_id']
    .count()
    .reset_index(name='bookings')
)
pivot_quarterly = quarterly_device.pivot(
    index='year_quarter', columns='device_type_used', values='bookings'
).fillna(0)
print('Q9 - Quarterly bookings by device:')
print(pivot_quarterly)

fig, ax = plt.subplots(figsize=(14, 6))
for device in pivot_quarterly.columns:
    ax.plot(pivot_quarterly.index.astype(str), pivot_quarterly[device],
            marker='o', linewidth=2, label=device)
ax.set_title('Quarterly Booking Trends by Device Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Year-Quarter')
ax.set_ylabel('Number of Bookings')
ax.legend(title='Device Type')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('q9_quarterly_device_trends.png', dpi=150)
plt.close()
print('  Saved: q9_quarterly_device_trends.png')
print()

# Q10
sessions_df = sessions.copy()
sessions_df['date'] = pd.to_datetime(sessions_df['search_time'].dt.date)
sessions_df['month'] = sessions_df['search_time'].dt.month
sessions_df['month_name'] = sessions_df['search_time'].dt.strftime('%B')
sessions_df['day_of_week'] = sessions_df['search_time'].dt.day_name()
sessions_df['has_booking'] = sessions_df['booking_id'].notna().astype(int)

monthly_obsr = (
    sessions_df.groupby(['month', 'month_name'])
    .apply(lambda g: g['has_booking'].sum() / len(g))
    .reset_index(name='oBSR')
    .sort_values('month')
)
print('Q10 - oBSR by month:')
for _, row in monthly_obsr.iterrows():
    print(f"  {row['month_name']:12s}: {row['oBSR']*100:.2f}%")

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(monthly_obsr['month_name'], monthly_obsr['oBSR'],
       color=sns.color_palette('Blues_d', 12))
ax.set_title('Average oBSR by Month of Year', fontsize=13, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('oBSR (Bookings / Searches)')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('q10_obsr_by_month.png', dpi=150)
plt.close()
print('  Saved: q10_obsr_by_month.png')

day_order_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_obsr = (
    sessions_df.groupby('day_of_week')
    .apply(lambda g: g['has_booking'].sum() / len(g))
    .reindex(day_order_week)
    .reset_index(name='oBSR')
)
print('\noBSR by day of week:')
for _, row in daily_obsr.iterrows():
    print(f"  {row['day_of_week']:12s}: {row['oBSR']*100:.2f}%")

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(daily_obsr['day_of_week'], daily_obsr['oBSR'],
       color=sns.color_palette('Oranges_d', 7))
ax.set_title('Average oBSR by Day of Week', fontsize=13, fontweight='bold')
ax.set_xlabel('Day of Week')
ax.set_ylabel('oBSR (Bookings / Searches)')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
plt.tight_layout()
plt.savefig('q10_obsr_by_day.png', dpi=150)
plt.close()
print('  Saved: q10_obsr_by_day.png')

daily_ts = (
    sessions_df.groupby('date')
    .apply(lambda g: g['has_booking'].sum() / len(g))
    .reset_index(name='oBSR')
    .sort_values('date')
)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily_ts['date'], daily_ts['oBSR'], linewidth=1.2, color='steelblue', alpha=0.8, label='Daily oBSR')
rolling_avg = daily_ts.set_index('date')['oBSR'].rolling(window=7, min_periods=1).mean()
ax.plot(rolling_avg.index, rolling_avg.values, linewidth=2, color='tomato', label='7-day rolling avg')
ax.set_title('oBSR Time Series (Daily)', fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('oBSR (Bookings / Searches)')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
ax.legend()
plt.tight_layout()
plt.savefig('q10_obsr_time_series.png', dpi=150)
plt.close()
print('  Saved: q10_obsr_time_series.png')

print('\nDone.')
