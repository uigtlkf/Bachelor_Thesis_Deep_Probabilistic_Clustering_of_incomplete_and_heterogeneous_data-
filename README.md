# Bachelorarbeit - Deep Probabilistic Clustering for Heterogeneous and Incomplete Data

# Project Structure:

IDEC.py: Defines the IDEC class, an Improved Deep Embedded Clustering model. It uses an encoder, decoder, and clustering layer to perform clustering and data reconstruction.


missingness_evaluation.py: Contains functions to simulate missing data (generate_missing_data) and evaluate different patterns of missingness using mechanisms such as MCAR (Missing Completely at Random) , MNAR (Missing Not at Random) and MAR (Missing At Random).


sampling.py: Implements the Sampling layer used in VAEs to generate latent variables by sampling from a normal distribution, given the mean and log variance.


MoE/: Contains the implementation of Mixture of Experts (MoE) VAE model and scripts for training them (moe_vae.py, train_moe.py, train_IDEC_moe.py).


MoPoE/: Includes the Mixture of Products of Experts (MoPoE) VAE model and scripts for their training (mopoe_vae.py, train_mopoe.py, train_IDEC_mopoe.py).


PoE/: Holds the Product of Experts (PoE) VAE model and training scripts (poe_vae.py, train_poe.py, train_IDEC_poe.py).

# How to run the script:

1. Clone the Repository:

cd `<repository-directory>`

git clone `<repository-link>`

2. Create a Virtual Environment:

python -m venv venv

source venv/bin/activate

To activate the virtual environment on Windows, use the following command:

venv\Scripts\activate

3. Install Dependencies:

pip install -r requirements.txt

4. Run the Training Script:

python -m src.MoPoE.train_IDEC_mopoe

python -m src.MoE.train_IDEC_moe

python -m src.PoE.train_IDEC_poe