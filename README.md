![brain_logo](https://github.com/SummerAnn/STR_SEI_Genomics/assets/107574104/870347b5-b00c-4927-be18-d1aa7a38c7b1)

## Genomics Project with Machine Learning and SEI Framework

![Genomics Project](https://img.shields.io/badge/Genomics-Machine%20Learning-blue)

Welcome to our Genomics project repository! This project combines the power of machine learning with the SEI (Sequence-Extraction-Inference) framework to leverage genetic sequence information and short tandem repeat (STR) analysis for disease classification.

Short tandem repeat (STR) analysis is a standard molecular biology method used to compare allele repeats at specific loci in DNA between two or more samples. A short tandem repeat is a microsatellite with repeat units that are 2 to 7 base pairs in length, with the number of repeats varying among individuals, making STRs effective for human identification purposes. This method differs from restriction fragment length polymorphism analysis (RFLP) since STR analysis does not cut the DNA with restriction enzymes. Instead, polymerase chain reaction (PCR) is employed to discover the lengths of the short tandem repeats based on the size of the PCR product.

We are investigating STR using the Sei framework. Sei is a framework for predicting sequence regulatory activities and applying sequence information to human genetics data. Sei is the deep learning tool for the predictions for 21,907 chromatin profiles. 

Sequence class label	Sequence class name	Rank by size	Group

PC1	Polycomb / Heterochromatin	0	PC
L1	Low signal	1	L
TN1	Transcription	2	TN
TN2	Transcription	3	TN
L2	Low signal	4	L
E1	Stem cell	5	E
E2	Multi-tissue	6	E
E3	Brain / Melanocyte	7	E
L3	Low signal	8	L
E4	Multi-tissue	9	E
TF1	NANOG / FOXA1	10	TF
HET1	Heterochromatin	11	HET
E5	B-cell-like	12	E
E6	Weak epithelial	13	E
TF2	CEBPB	14	TF
PC2	Weak Polycomb	15	PC
E7	Monocyte / Macrophage	16	E
E8	Weak multi-tissue	17	E
L4	Low signal	18	L
TF3	FOXA1 / AR / ESR1	19	TF
PC3	Polycomb	20	PC
TN3	Transcription	21	TN
L5	Low signal	22	L
HET2	Heterochromatin	23	HET
L6	Low signal	24	L
P	Promoter	25	P
E9	Liver / Intestine	26	E
CTCF	CTCF-Cohesin	27	CTCF
TN4	Transcription	28	TN
HET3	Heterochromatin	29	HET
E10	Brain	30	E
TF4	OTX2	31	TF
HET4	Heterochromatin	32	HET
L7	Low signal	33	L
PC4	Polycomb / Bivalent stem cell Enh	34	PC
HET5	Centromere	35	HET
E11	T-cell	36	E
TF5	AR	37	TF
E12	Erythroblast-like	38	E
HET6	Centromere	39	HET

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Figures](#figures)
- [Contributors](#contributing)
- [License](#license)

## Introduction

Genomics is at the forefront of modern medicine, and leveraging machine learning techniques can significantly enhance our ability to understand and classify diseases based on genetic data. In this project, we employ the SEI framework to process genetic sequence information and short tandem repeats, ultimately building a robust disease classification model.



## Features

- Utilizes genetic sequence data
- Implements Short Tandem Repeat (STR) analysis
- Applies machine learning algorithms for disease classification
- SEI framework for efficient data processing

## Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

- Python (>=3.6)
- Pip (Python package manager)

### Installation

1. Clone this repository to your local machine.
```bash
git clone https://github.com/yourusername/genomics-ml-sei.git
```
## Usage
Describe how to use your project here. Include any necessary steps, commands, or examples for running the machine learning model and utilizing the SEI framework for disease classification.

## Data
Explain the source and format of your genetic data. Provide details on where to access or obtain the data if applicable. You can also include a brief description of data preprocessing steps.

## Methodology
Describe the SEI framework and the machine learning algorithms employed in this project. Include relevant links to external resources and libraries used.

## Results
Share the outcomes and insights gained from your project. Include performance metrics, visualizations, or any other relevant results. Discuss the significance of your findings.

## Figures

## Contributors

## License
Add license info here.

## Thank you!
Thank you for your interest in our Genomics project! If you have any questions or suggestions, please feel free to reach out to us. We look forward to advancing genomics research together with the power of machine learning and the SEI framework.
