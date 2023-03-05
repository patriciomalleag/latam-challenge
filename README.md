# Latam Challenge

This repository contains the solution to the Latam Challenge, which consists in predicting the probability of delay of the flights that take off from the airport of Santiago de Chile (SCL) using data from 2017.

## Requirements

To replicate this repository, you will need to have the following software installed on your computer:

- Anaconda Distribution: https://www.anaconda.com/products/distribution

## Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/patriciomalleag/latam-challenge.git
    ```

2. Create a new Anaconda environment:

    ```bash
    conda create --name latam-challenge python=3.10 -y
    ```

3. Activate the environment:

    ```bash
    conda activate latam-challenge
    ```

4. Install the necessary packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use this repository, simply run the Jupyter Notebook "solution.ipynb" with your Python environment activated:

```bash
jupyter notebook solution.ipynb
```

## Releases

- Release v1.1.0-general-data-preparation 
  * Feature: model-training 
  * Feature: model-data-preparation (2023-03-05T13:43)
  * Feature: data-analysis-delay-rate (2023-03-04T19:02)

- Release v1.0.0-general-data-preparation (2023-03-04T15:43)
  * Feature: data-preparation (2023-03-04T15:34)
  * Feature: synthetic-features (2023-03-03T22:04)
  * Feature: data-exploration (2023-03-03T20:53)

## Additional Information
If you encounter any issues while replicating this repository, please feel free to contact me.

## Credits
This repository was created by [Patricio Mallea](https://www.linkedin.com/in/patriciomallea/).