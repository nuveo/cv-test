# Description

Picture yourself strolling through your local, open-air market... What do you see? What do you smell? What will you make for dinner tonight?

If you're in Northern California, you'll be walking past the inevitable bushels of leafy greens, spiked with dark purple kale and the bright pinks and yellows of chard. Across the world in South Korea, mounds of bright red kimchi greet you, while the smell of the sea draws your attention to squids squirming nearby. India’s market is perhaps the most colorful, awash in the rich hues and aromas of dozens of spices: turmeric, star anise, poppy seeds, and garam masala as far as the eye can see. In fact, there is a common assumption that considers a strong relationship between geographic and cultural associations with local foods.

# Objective

This test asks you to predict the category of a dish's cuisine given a list of its ingredients. The category is defined by the region where the recipe come from, following the list:

```
'brazilian', 'british', 'cajun_creole', 'chinese', 'filipino',
'french','greek', 'indian', 'irish', 'italian', 'jamaican',
'japanese','korean','mexican', 'moroccan', 'russian',
'southern_us', 'spanish', 'thai', 'vietnamese'
```

# Important details

- The dataset was split in order to have unseen data for analysis. We took 5% of the total data (randomly). This is given to the candidate in the folder `TestSet`, except for the `cuisine` information that is missing.
- The accepted answer will be the same file found at `TestSet`, however with the `cuisine` information provided by the prediction model. Thus, the candidate will provide the same structure found at the `TrainingSet`
- This test does not require a defined set of algorithms to be used. The candidate is free to choose any kind of data processing pipeline to reach the best answer.
