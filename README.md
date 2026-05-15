# Travel Aggregator Analysis - MyNextBooking

Exploratory data analysis of booking and search data from MyNextBooking, a travel aggregator platform. This was my certification project for the Edureka Data Science and Machine Learning program.

The dataset covers flights booked across three OTA services (GOIBIBO, YATRA, MMT) — 339 bookings and 1,497 session-level search records.

---

## Questions Explored

1. How many distinct bookings, sessions, and searches are there?
2. How many sessions resulted in more than one booking?
3. Which day of the week sees the highest number of bookings?
4. What is the total booking count and Gross Booking Value (GBV) per service?
5. What is the most booked route among customers who book more than once?
6. Which departure cities have the longest average advance booking window?
7. What are the correlations between numerical features?
8. Which device type is most used per service?
9. How have bookings trended quarterly across device types?
10. What is the Booking-to-Search Ratio (oBSR) by month and day of week?

---

## Key Findings

- **Thursday** had the highest number of bookings (65 out of 339)
- **GOIBIBO** led all services with 186 bookings and ₹58.97L in gross booking value
- Most repeated route among loyal customers: **Gurgaon → Roissy-en-France** (5 bookings)
- Strongest numerical correlation: **INR Amount & distance_km** (r = 0.62)
- oBSR peaked in **June (39.4%)** and was lowest in **July (16.3%)**
- **Sunday** had the best search-to-booking conversion rate (33.6%)

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook travel_aggregator_analysis.ipynb
```

---

## Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · Jupyter
