# Results summary

## Data
- 431,708 reviews across 9,912 businesses
- Star ratings: 1 star 17.4%, 2 star 8.1%, 3 star 9.0%, 4 star 18.7%, 5 star 46.9%

## VADER fix (checked against the customers' own ratings)
| | Before (cleaned text) | After (original text) |
|---|---|---|
| 1-2 star reviews labelled negative | 40.0% | 53.1% |
| 1-2 star reviews labelled positive | 56.3% | 43.9% |
| Agreement with star rating (neg/neutral/pos) | 74.4% | 77.5% |
| Reviews labelled positive overall | 86.6% | 82.6% |

## Most negative categories (share of 1-2 star reviews; all reviews 25.5%)
- Television Service Providers: 84.3% of 510 reviews
- University Housing: 77.9% of 535 reviews
- Property Management: 69.9% of 1,418 reviews
- Internet Service Providers: 68.5% of 834 reviews
- Apartments: 66.2% of 2,905 reviews
- Banks & Credit Unions: 64.2% of 982 reviews
- Real Estate: 61.7% of 4,933 reviews
- Hospitals: 60.2% of 1,059 reviews

## Star rating predictor (test set: 86,342 reviews)
| Model | Accuracy | Within 1 star | Mean error (stars) | Flags 1-2 star: recall | Flags 1-2 star: precision |
|---|---|---|---|---|---|
| Always guess the most common rating | 46.9% | 65.5% | 1.31 | 0.0% | 0.0% |
| Old setup: cleaned text, top 3,000 words | 69.2% | 93.0% | 0.42 | 84.3% | 84.9% |
| New setup: original text, words + 2-word phrases | 72.2% | 95.8% | 0.34 | 87.9% | 88.2% |

Recall by actual rating (new setup): 1 star 87.0%, 2 star 34.9%, 3 star 40.6%, 4 star 46.4%, 5 star 89.4%

Strongest 1-star phrases: worst, horrible, one star, terrible, rude, awful, disgusting, zero stars, not, never, will never, very disappointed

Strongest 5-star phrases: amazing, great, delicious, best, awesome, excellent, love, perfect, fantastic, not only, so good, wonderful

## Complaint themes (110,081 reviews rated 1-2 stars; 5 topics chosen by NPMI coherence)
- T1 (41.7%): food, order, good, eat, restaurant, service, table, taste, chicken, bad
- T2 (24.0%): customer, service, wait, store, manager, people, order, minute, employee, rude
- T3 (19.6%): service, car, company, hour, phone, work, appointment, pay, week, charge
- T4 (9.2%): room, dog, clean, stay, hotel, door, car, work, water, night
- T5 (5.6%): nail, hair, cut, apartment, bad, pay, leave, good, color, salon

Largest theme in each sector:
- Restaurants & Food: T1: food, order, good (67.9%)
- Home Services: T3: service, car, company (54.2%)
- Automotive: T3: service, car, company (73.1%)
- Health & Medical: T3: service, car, company (75.5%)
- Shopping: T2: customer, service, wait (60.6%)
- Beauty & Spas: T5: nail, hair, cut (71.8%)
- Hotels & Travel: T4: room, dog, clean (67.5%)

## Closure backtest (signals from reviews before 2019-01-01)
- 3,628 businesses open at the cutoff; 333 closed by the snapshot (9.2%)
- Cross-validated AUC: all signals 0.624, ratings only 0.549
- Of the 15 businesses flagged highest risk, 5 closed

| Tier | Businesses | Closed | Closure rate | Lift vs average |
|---|---|---|---|---|
| High risk | 363 | 67 | 18.5% | 2.01x |
| Elevated | 726 | 93 | 12.8% | 1.40x |
| Low | 2,539 | 173 | 6.8% | 0.74x |

Standardised weights (positive = more likely to close):
- Restaurant: +0.48
- Rating (Bayesian average): -0.25
- Years since first review: -0.22
- Chain: -0.18
- Reviews in the last 12 months (log): -0.11
- Share of 1-2 star reviews, last 12 months: -0.05
- Review volume trend, last 12 vs prior 12 months: +0.01
- Rating trend, last 12 vs prior 12 months: +0.01

## Revenue illustration (open independent restaurants)
- 1,097 restaurants; average rating 3.79; 512 below average
- Median upside for those below average: $12,785 a year (assumes 5.0% per star and $800,000 revenue)
