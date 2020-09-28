# Description

As I scurried across the candlelit chamber, manuscripts in hand, I thought I'd made it. Nothing would be able to hurt me anymore. Little did I know there was one last fright lurking around the corner.

DING! My phone pinged me with a disturbing notification. It was Antonio, the scariest of Nuveo reviewer, sharing news of another data leak.

"ph’nglui mglw’nafh Cthulhu R’lyeh wgah’nagl fhtagn!" I cried as I clumsily dropped my crate of unbound, spooky books. Pages scattered across the chamber floor. How will I ever figure out how to put them back together according to the authors who wrote them? Or are they lost, forevermore? Wait, I thought... I know, machine learning!

Setting the `train` and `test` files as formatted using samples of each authors texts, using the stands such as Edgar Allan Poe (`EAP`), Mary Shelley (`MWS`) and HP Lovecraft (`HPL`), I've noticed that this problem can be solved.

Each line is of these files as composed by two columns: one with text id which defines the text sample, one with the text fragment itself and, lastly, the final column with the authorship identification (`EAP`, `HPL` or `MWS`). Here are some examples:

```
id27839	Now the direct inference from this is that the machine is not a pure machine.	EAP
id02834	That object no larger than a good sized rat and quaintly called by the townspeople "Brown Jenkin" seemed to have been the fruit of a remarkable case of sympathetic herd delusion, for in no less than eleven persons had testified to glimpsing it.	HPL
id21889	Harsh but impressive polysyllables had replaced the customary mixture of bad Spanish and worse English, and of these only the oft repeated cry "Huitzilopotchli" seemed in the least familiar.	HPL
id01542	Then I wandered from the fancies of others and formed affections and intimacies with the aerial creations of my own brain but still clinging to reality I gave a name to these conceptions and nursed them in the hope of realization.	MWS
id23283	Then a rushing revival of soul and a successful effort to move.	EAP
id00082	Of the dungeons there had been strange things narrated fables I had always deemed them but yet strange, and too ghastly to repeat, save in a whisper.	EAP

```

For evaluation purposes, the `test` dataset does not prosent the authorship information (`EAP`, `HPL` or `MWS`) in the last column. Therefore, the `train` data is the full source of information for this test.

# Objective

In this test, you're challenged to predict the authorship of excerpts from horror stories by Edgar Allan Poe, Mary Shelley, and HP Lovecraft. 

# Important details

- The dataset was split in order to have unseen data for analysis. We took 15% of the total data (randomly)
- Replicate the data format for submission, i.e. the answer must be provided as a CSV file with the detect authorship in the last column and the rest as usual (i.e. text id and text information in the first and second columns respectively), similarly to what is provided in the `TrainingSet` file
- The `TestSet` will be used for evaluation, therefore the candidate must fullfiled the last column with the predicted classes (`EAP`, `HPL` or `MWS`)
- This test does not require a defined set of algorithms to be used. The candidate is free to choose any kind of data processing pipeline to reach the best answer.

# How to Run

The SMS Ham-Spam test is described in `SMSSpamDetection.ipynb` file. All dataset modifications and ideas over the NLP tasks are described there. If you want to test this approach, you only need to:

- Download the provided [dataset](https://drive.google.com/file/d/1LhH_5ULfyrobD60SZqIfoI56eV3HuDNI/view) from NUVEO. 
- Change both train/test path inside the notebook.
- install the libraries by doing `pip install -r requirements.txt` 