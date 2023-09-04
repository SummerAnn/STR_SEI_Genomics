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
1. Short Tandem Repeats (STRs): Short Tandem Repeats (STRs), also known as microsatellites or simple sequence repeats, are short sequences of DNA that consist of repeating units of 2 to 6 base pairs. These sequences are found throughout the human genome and in the genomes of many other organisms. 
The term "tandem repeat" refers to the fact that the repeated sequences are located adjacent to each other, like beads on a string. For example, a common STR might have the sequence "ACACACACACAC."
Key points about STRs:
**Variable Length**: The number of repeats in an STR can vary among individuals. This variation is what makes STRs valuable for genetic profiling and forensic analysis.
**Genetic Markers**: STRs are used as genetic markers because they are highly polymorphic, meaning they vary significantly from person to person. This makes them useful for tasks like DNA fingerprinting.
**Mutation Prone**: STRs are prone to mutation, which can lead to changes in the number of repeats over generations. This makes them useful for studying evolution and population genetics.
**Applications**: STR analysis is widely used in fields like forensic science, paternity testing, and population genetics. They are also used in research to map genes associated with various genetic disorders.
In summary, STRs are short sequences of DNA with repeating units that are highly variable among individuals, making them important tools in genetics, forensics, and population studies.
2. Why Machine Learning is valuable for analyzing STRs:
**Pattern Recognition**: ML algorithms excel at recognizing patterns in data. In the context of STR analysis, ML can automatically identify and classify the repetitive patterns of DNA sequences, which can be challenging and time-consuming for humans to do manually.
**Classification**: ML can classify STR sequences into different categories or allele lengths. This is crucial in applications like forensic genetics and genetic disease diagnosis, where distinguishing between different allele lengths is essential.
Predictive Modeling: ML models can predict the likelihood of certain alleles or patterns of STRs occurring in an individual's genome based on training data. This predictive capability is useful for genetic profiling and assessing genetic risk factors for diseases.
**Population Genetics**: ML can help analyze large datasets of STR profiles from diverse populations. It can uncover population-specific patterns and genetic variations, aiding in understanding human migration, genetic diversity, and population history.
**Mutation Prediction**: ML algorithms can be trained to predict the likelihood of STR mutations. This is valuable in studying the evolution of STRs over generations and their role in genetic diseases caused by repeat expansions or contractions.
**Feature Extraction**: ML techniques can automatically extract relevant features or characteristics from STR data, which can be used as input for downstream analyses or for training more complex models.
**Data Integration**: In bioinformatics, ML can be used to integrate STR data with other omics data (e.g., genomics, transcriptomics, proteomics) to gain a comprehensive understanding of how STR variations relate to gene expression, protein function, and disease mechanisms.
**Automation and Speed**: ML can automate the analysis of large-scale STR datasets, significantly speeding up the process and reducing the risk of human error.
**Scalability**: ML techniques are scalable, making it possible to analyze vast amounts of genomic data, which is increasingly common in modern genetics and genomics research.
In summary, Machine Learning is valuable for STR analysis because it can automate, accelerate, and enhance the interpretation of STR data, enabling researchers and practitioners to extract meaningful insights, make accurate predictions, and better understand the genetic variations and implications of STRs in various fields of genetics and genomics.
3. Introduce the SEI framework and its relevance to bioinformatics projects.
4. Provide an overview of the tools and technologies you'll be using, such as Python, bedtools, and downstream/upstream analysis.



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
