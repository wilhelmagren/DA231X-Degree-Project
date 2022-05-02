# DA231X-Degree-Project

## Project description
In collaboration with Karolinska Institutet and Stockholm University the [SLEMEG dataset](https://www.su.se/english/research/research-projects/slemeg-an-meg-study-on-the-effects-of-insufficient-sleep-on-emotional-and-attentional-processes) has been used to investigate contrastive self-supervised learning paradigms for automatic learning and extraction of features on MEG data. In particular, the SimCLR framework proposed by Hinton et al. 2020 has been utilized and studied to propose 3 models that aim to learn informative representations on the MEG that to be used in classification and regression downstream tasks. The dataset includes resting state recordings from 33 subjects where they once were present with normal sleep schedules and once when permitted only 4 hours of sleep per night, being prone to partial sleep deprivation (psd). Thus, the overall goal and aim is for the models to encode a representation that distinguishes between control and psd samples, that also manages to generalize sufficiently well.

### Research question and objectives
"*How does the choice of self-supervised learning approach impact the learned latent space representation considering its informative properties relating to subject labels of the MEG data?*"

1. Visually inspect the extracted features by means of the non-linear dimensionality reduction technique t-SNE and investigate if there exists distinct clusters according to subject labels.
2. Apply k-Means to the extracted features to determine the cluster robustness with a k-fold cross validation scheme and evaluate the clusters with silhouette coefficient metric.
3. Utilize the learned latent space representation to perform classification and regression on the corresponding subject labels to determine the SSL methods transfer learning capabilities in a downstream task context.

## Neurocode
![Neurocode logo](/images/neurocode.png)
This is a small library I wrote to streamline working with the provided SLEMEG dataset (but works for other EEG/MEG datasets as well). It features loading data, preprocessing and removing artifacts, pretext tasks, downstream tasks, and training and evaluation of models! Inspiration was taken from the library [braindecode](https://braindecode.org/) for this project, but directly using that library did not work with the SLEMEG dataset. Hence, much of the code is inspired by that of braindecode but tailored for the provided SLEMEG project. It is dependent on the following libraries: <br> > [numpy](https://numpy.org/), [mne-python](https://mne.tools/stable/index.html), [sci-kit learn](https://scikit-learn.org/stable/), [pytorch](https://pytorch.org/).

## Administrative
Degree Project in Computer Science and Engineering at KTH Royal Institute of Technology, advanced level 30 credits. Yields a M.Sc.Eng degree as part of the Information- and Communication Technology program CINTE. The contents and learning outcomes of the course are as follows:

*"Before the degree project course is started, the student should identify an appropriate degree project assignment and formulate a project proposal, so that this can be approved by the examiner. The assignment should be chosen so that it implies a natural progression of the knowledge and skills that have been acquired within the education and in a possible specialization within the education."*

- The student writes an individual plan for the project, in which the problem description/assignment and preconditions for the implementation of the work are specified.
- The student carries out an in-depth-pre-study including discussions of method choice and theoretical backgroudn with a literature a literature study that is reported as part of a draft to a preliminary version of the written degree project report.
- The student independently carries out a degree project, where knowledge and methods from the education are applied.
- The student plans and carries out an oral presentation and defence of his or her degree project.
- The student carries out an oral and written review of another second-cycle degree project.
- The student writes and presents a written degree project report, where the student for and discusses own conclusions in the degree project and the knowledge and the arguments that underpin them.
- The student carries out a self-evaluation of the degree project according to established model.

The examination of the course is separated in three different parts. They all sum up to the total 30 credits which are expected of the project course.
- PRO1, 7.5 credits: an individual plan for degree project, a pre-study, a discussion of method, and a literature study.
- PRO2, 15.0 credits: a written report with abstract in both Swedish and English, a self-assesment report.
- PRO3, 7.5 credits: an oral presentation, a written and oral review (public discussion) of another student's second-cycle degree project, the final version of the report.

Active attendance at two oral presentations of degree projects for second-cycle studies is also required to pass the course. This will most likely be done during PRO3. All examination parts should be approved within a year from the start of the degree project. Otherwise, the degree project will be ended with a failed grade, unless special circumstances apply.

Links to course rooms are found below:
- [DA231X Degree Project in Computer Science and Engineering, Second Cycle 30.0 credits](https://www.kth.se/student/kurser/kurs/DA231X?l=en)
- [Degree Projects at EECS, 2022](https://canvas.kth.se/courses/33514)

### Contact and license
Author: Wilhelm Ågren, wagren@kth.se
<br>License: GNU General Public License v3.0
