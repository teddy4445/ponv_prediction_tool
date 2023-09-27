# Predicting Early and Delayed Postoperative Nausea And Vomiting Using Machine Learning: A Model Development and Validation Study

## Abstract
Postoperative nausea and vomiting (PONV) is one of the most frequently occurring complications in patients undergoing surgery with general anesthesia. It is one of the most reported causes of distress and dissatisfaction among surgical patients in the early postoperative hours. Current PONV prediction tools are used to reduce PONV, producing unsatisfactory results. In this study, we aimed to develop prognostic PONV prediction tools for early and delayed PONV that provide satisfactory prediction performance. In this model development and validation study, we used post-prospective data of adult patients admitted to the post-anesthesia care unit who had undergone surgical procedures conducted under general anesthesia at The Sheba Medical Center, Israel, between 1 September 2018 and 1 September 2023. Using this data, we developed an ensemble model of machine learning algorithms which is trained on roughly 35 thousand patients' data. We use the \textit{k}-fold cross-validation method followed by a split to train and test the data that optimally preserve the socio-demographic features of the patients such as age, gender, and smoking habits using the Bee Colony algorithm. We identified data for 54672 patients, with 2706 (4.93\%) and 8128 (14.98\%) patients suffering from early and delayed PONV, respectively. We show that using this data, the obtained PONV prediction tools are able to correctly predict early and delayed PONV 84.0\% and 77.3\% of the time which outperformed the second-best PONV prediction tool (Koivuranta score) by 13.4\% and 12.9\%, respectively. Moreover, a feature importance analysis reveals that the proposed prediction tools align with previous clinical knowledge, assuring its evaluability. An accurate PONV prediction tool can be used by clinicians in general and anesthesiologists, in particular, to evaluate a surgery plan and provide better clinical care for patients. Further evaluation of the proposed PONV prediction tool should be extended to an automatic suggestion tool. \\

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
        title={Predicting Early and Delayed Postoperative Nausea And Vomiting Using Machine Learning: A Model Development and Validation Study},
        author={Glebov, M. and Lazebnik, T. and Bunimovich, S. and Orkin, B.},
        journal={<<TBD>>},
        year={2023}
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
* Teddy Lazebnik - [email](mailto:t.lazebnik@ucl.ac.uk) | [LinkedInֿ](https://www.linkedin.com/in/teddy-lazebnik/)

