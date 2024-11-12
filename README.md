# Predicting Postoperative Nausea And Vomiting Using Machine Learning: A Model Development and Validation Study

## Abstract

Background:
Postoperative nausea and vomiting (PONV) is a frequently observed complication in patients undergoing surgery under general anesthesia. Moreover, it is a frequent cause of distress and dissatisfaction in the early postoperative period. Currently, the classical scores used for predicting PONV have not yielded satisfactory results. Therefore, prognostic models for the prediction of early and delayed PONV were developed in this study to achieve satisfactory predictive performance.

Methods:
The retrospective data of inpatient adult patients admitted to the post-anesthesia care unit after undergoing surgical procedures under general anesthesia at the Sheba Medical Center, Israel, between September 1, 2018, and September 1, 2023, were used in this study. An ensemble model of machine-learning algorithms trained on the data of 35003 patients was developed. The k-fold cross-validation method was used followed by splitting the data to train and test sets that optimally preserve the sociodemographic features of the patients.

Findings:
Among the 35003 patients, early and delayed PONV were observed in 1340 (3.82%) and 6582 (18.80%) patients, respectively. The proposed PONV prediction models correctly predicted early and delayed PONV in 83.6% and 74.8% of cases, respectively, outperforming the second-best PONV prediction score (Koivuranta score) by 13.0% and 10.4%, respectively. Feature importance analysis revealed that the performance of the proposed prediction tools aligned with previous clinical knowledge, indicating their utility.

Interpretation:
The machine learning-based models developed in this study enabled improved PONV prediction, thereby facilitating personalized care and improved patient outcomes.

Funding:
This study represents an independent research project funded by Ariel University and the Holon Institute of Technology (grant number RA2300000519).

## Table of contents
1. [Code usage](#code_usage)
2. [How to cite](#how_to_cite)
3. [Dependencies](#dependencies)
4. [Contributing](#contributing)
5. [Contact](#contact)

<a name="code_usage"/>

## Code usage
### Run the experiments shown in the paper:
1. Clone the repo 
2. Install the `requirements.txt` file.
3. run the project  

<a name="how_to_cite"/>

## How to cite
Please cite the SciMED work if you compare, use, or build on it:
```
@article{glebov2023ponv,
        title={Predicting Postoperative Nausea And Vomiting Using Machine Learning: A Model Development and Validation Study},
        author={Glebov, M. and Lazebnik, T. and Orkin, B. and Berkenstadt, H. and Bunimovich, S. },
        journal={<<TBD>>},
        year={2025}
}
```

<a name="dependencies"/>

## Dependencies 
1. pandas 
2. numpy 
3. matplotlib 
4. seaborn 
5. scikit-learn 
6. scipy 
7. TPOT

<a name="contributing"/>

## Contributing
We would love you to contribute to this project, pull requests are very welcome! Please send us an email with your suggestions or requests...

<a name="contact"/>

## Contact
* Teddy Lazebnik - [email](mailto:lazebnik.teddy@gmail.com) | [LinkedInֿ](https://www.linkedin.com/in/teddy-lazebnik/)

