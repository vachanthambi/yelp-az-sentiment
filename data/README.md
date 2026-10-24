# Data

The raw data isn't in this repo (it's several GB, and Yelp's terms don't allow
redistributing it).

1. Download the [Yelp Open Dataset](https://www.yelp.com/dataset) and accept its terms.
2. Convert the business and review JSON files to CSV, keeping their field names
   (`business_id`, `name`, `city`, `state`, `categories`, `stars`,
   `review_count`, `is_open` for businesses; `review_id`, `business_id`,
   `stars`, `text`, `date` for reviews).
3. Save them here as:
   - `data/yelp_academic_dataset_business.csv`
   - `data/yelp_academic_dataset_review.csv`

`is_open` matters: step 06 uses it as the outcome for the closure backtest.

The pipeline writes its intermediate files here too (`az_reviews.csv`,
`reviews_preprocessed.csv`). All CSV and JSON files in this folder are
git-ignored.
