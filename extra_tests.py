#!/usr/bin/env python3
"""
Additional RYS evaluation probe to test for model degradation.

Pulls questions from cached HuggingFace datasets (BBH) plus 56 question
custom subset of TinyMMLU benchmark. All questions produce short outputs and
are objectively scorable without a judge model.

Usage:
    python extra_tests.py \
        --model /path/to/model.gguf \
        --llama-server /path/to/llama-server \
        --tmpdir /dev/shm/rys \
        --port 8099
        --output results.jsonl
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset
from ls_utils import wait_for_server, start_server, stop_server, dump_server_log, query_model


# Server config
DEFAULT_PORT = 8099


TINYMMLU_QUESTIONS = [
    # nutrition
    {
        "prompt": (
            "According to trials on diabetes prevention, high-risk individuals "
            "can reduce their risk to develop diabetes by doing the following:\n"
            "Options:\n"
            "(a) Eating a very low carbohydrate diet\n"
            "(b) Consuming a diet high in monounsaturated fats\n"
            "(c) Losing 5-7% of body weight through a hypocaloric low fat diet and 30 minutes of daily activity\n"
            "(d) Initiating metformin 850 mg BID and practicing daily vigorous exercise\n"
        ),
        "answer": "c",
    },
    # clinical_knowledge
    {
        "prompt": (
            "Which of the following is true regarding the complications of a "
            "large goitre?\n"
            "Options:\n"
            "(a) It causes stridor due to compression of the oesophagus\n"
            "(b) Dysphagia is an indication for surgery\n"
            "(c) It causes hoarseness due to compression of the trachea\n"
            "(d) It causes a more florid thyrotoxicosis than a smaller goitre\n"
        ),
        "answer": "b",
    },
    # nutrition
    {
        "prompt": (
            "Doses of substances used in animal toxicity studies are usually "
            "many times greater than those experienced by people because:\n"
            "Options:\n"
            "(a) Animals are generally much less sensitive than people\n"
            "(b) Such tests can detect only very large risks because of the fact that only small numbers of animals can be used\n"
            "(c) It is difficult in such studies to determine whether the observed toxic effects were caused by the substance\n"
            "(d) It is the only way to ensure results are applicable to people\n"
        ),
        "answer": "b",
    },
    # professional_medicine
    {
        "prompt": (
            "A 17-year-old girl is brought to the physician by her mother "
            "because she has not had a menstrual period for 6 months. The "
            "patient is unconcerned about the lack of menses. Menarche "
            "occurred at the age of 12 years, and menses had occurred at "
            "regular 28-day intervals until they became irregular 1 year ago. "
            "She is a member of her high school gymnastics team. She appears "
            "emaciated. She is 163 cm (5 ft 4 in) tall and weighs 40 kg (88 "
            "lb); BMI is 15 kg/m2 . Her pulse is 54/min, and blood pressure is "
            "80/50 mm Hg. Which of the following is the most likely cause of "
            "this patient's amenorrhea?\n"
            "Options:\n"
            "(a) Hyperthyroidism\n"
            "(b) Hypogonadotropic hypogonadism\n"
            "(c) Hypothyroidism\n"
            "(d) Polycystic ovarian syndrome\n"
        ),
        "answer": "b",
    },
    # prehistory
    {
        "prompt": (
            "The Yang-shao culture gave way to the Lung-Shan sometime after:\n"
            "Options:\n"
            "(a) 6,000 B.P.\n"
            "(b) 5,000 B.P.\n"
            "(c) 4,000 B.P.\n"
            "(d) 3,000 B.P.\n"
        ),
        "answer": "b",
    },
    # philosophy
    {
        "prompt": (
            "Of the ways of escaping moral arguments that he discusses, Hare "
            "claims:\n"
            "Options:\n"
            "(a) none of them are successful.\n"
            "(b) some, but not all of them are successful.\n"
            "(c) all are entirely successful.\n"
            "(d) all might be successful, but at a price.\n"
        ),
        "answer": "d",
    },
    # professional_psychology
    {
        "prompt": (
            "An employee of large manufacturing firm finds the work dull, and "
            "pursues the job only for the money. The employee performs "
            "minimally, and is not motivated so produce highest quantity or "
            "better quality. According to F. Herzberg's two-factor theory, the "
            "best way 10 increase this worker's self-motivation would be to\n"
            "Options:\n"
            "(a) offer a raise and incentive pay for greater productivity\n"
            "(b) redesign the job to provide a challenge and a sense of accomplishment\n"
            "(c) increase opportunities for informed social interaction\n"
            "(d) provide the employee with a job tide carrying greater prestige\n"
        ),
        "answer": "b",
    },
    # professional_accounting
    {
        "prompt": (
            "In an e-commerce environment that requires that the information "
            "technology (IT) system be available on a continuous basis, more "
            "emphasis will be placed on which of the following aspects of the "
            "planning than in a traditional organization?\n"
            "Options:\n"
            "(a) Maintain appropriate written source documents so the data can be re-entered if it is lost or compromised\n"
            "(b) Maintain redundant systems for instant availability to assure the flow of transactions\n"
            "(c) Review additional expenses to obtain the required amount of business interruption insurance coverage for the organization\n"
            "(d) Assure that appropriate data backups are stored in an off-site location\n"
        ),
        "answer": "b",
    },
    # anatomy
    {
        "prompt": (
            "At which of the following locations does bile enter the digestive "
            "tract?\n"
            "Options:\n"
            "(a) Gastroesophageal sphincter\n"
            "(b) Duodenum\n"
            "(c) Ileocecum\n"
            "(d) Jejunum\n"
        ),
        "answer": "b",
    },
    # high_school_world_history
    {
        "prompt": (
            "This question refers to the following information. Is it not "
            "unity alone that can weld us into an effective force, capable of "
            "creating our own progress and making our valuable contribution to "
            "world peace? Which independent African state will claim that its "
            "financial structure and banking institutions are fully harnessed "
            "to its national development? Which will claim that its material "
            "resources and human energies are available for its own national "
            "aspirations? We are fast learning that political independence is "
            "not enough to rid us of the consequences of colonial rule. We "
            "have been too busy nursing our separate states to understand "
            "fully the basic need for union, rooted in common purpose, common "
            "planning and common endeavour. Ghana's president, Kwame Nkrumah, "
            "addressing the Organization of African Unity, 1963 The speaker in "
            "the passage above is espousing which of the following causes?\n"
            "Options:\n"
            "(a) Nationalism\n"
            "(b) Socialism\n"
            "(c) Pan-Africanism\n"
            "(d) Neocolonialism\n"
        ),
        "answer": "c",
    },
    # prehistory
    {
        "prompt": (
            "Monte Alban is located in the ________ and was built by the "
            "_________.\n"
            "Options:\n"
            "(a) Valley of Mexico; Aztec.\n"
            "(b) Valley of Oaxaca; Zapotec.\n"
            "(c) Amazon floodplain; Olmec.\n"
            "(d) Yucatán Peninsula; Maya.\n"
        ),
        "answer": "b",
    },
    # high_school_geography
    {
        "prompt": (
            "Which language family is most widely spoken in North America and "
            "Europe and includes Baltic, Celtic, Germanic, and Greek?\n"
            "Options:\n"
            "(a) Dravidian\n"
            "(b) Uralic-Altaic\n"
            "(c) Sino-Tibetan\n"
            "(d) Indo-European\n"
        ),
        "answer": "d",
    },
    # virology
    {
        "prompt": (
            "How does rubella cause foetal abnormalities?\n"
            "Options:\n"
            "(a) By crossing the placenta early in pregnancy and infecting the foetus\n"
            "(b) By only infecting the placenta\n"
            "(c) By inducing cytokines and chemokines in the mother\n"
            "(d) By raising the temperature of the mother and inducing an abnormal immune reaction to the foetus\n"
        ),
        "answer": "a",
    },
    # high_school_chemistry
    {
        "prompt": (
            "Silver metal, often amalgamated with mercury, is used to reduce "
            "substances to a desired oxidation state. If the silver metal "
            "amalgam cannot be used because of concerns about mercury, which "
            "of the following would be a reasonable and safe substitute?\n"
            "Options:\n"
            "(a) H+(aq)\n"
            "(b) Na(s)\n"
            "(c) Ca2+(aq)\n"
            "(d) Mg(s)\n"
        ),
        "answer": "d",
    },
    # professional_medicine
    {
        "prompt": (
            "A 52-year-old man is brought to the emergency department 30 "
            "minutes after he had an episode of chest pain radiating to his "
            "jaw while shoveling snow. His pulse is 80/min, and blood pressure "
            "is 130/70 mm Hg. The lungs are clear to auscultation. Cardiac "
            "examination shows an S4. While undergoing an ECG, the patient "
            "says that he feels the chest pain returning. The most appropriate "
            "immediate treatment is a drug with which of the following "
            "mechanisms of action?\n"
            "Options:\n"
            "(a) Increases cAMP concentration\n"
            "(b) Increases nitric oxide concentration\n"
            "(c) Inhibits potassium flux\n"
            "(d) Inhibits sodium flux\n"
        ),
        "answer": "b",
    },
    # high_school_chemistry
    {
        "prompt": (
            "A mechanism is a sequence of elementary reactions that add up to "
            "the overall reaction stoichiometry. A substance that is produced "
            "in one elementary reaction and consumed in another is called\n"
            "Options:\n"
            "(a) a catalyst\n"
            "(b) an intermediate\n"
            "(c) a reactant\n"
            "(d) a complex\n"
        ),
        "answer": "b",
    },
    # professional_medicine
    {
        "prompt": (
            "A 22-year-old woman comes to the office because of urticaria. "
            "This is her first episode of urticaria and it has occurred and "
            "then resolved several times in the past week. The history and "
            "physical examination disclose no abnormalities. Which of the "
            "following is the most appropriate course of action?\n"
            "Options:\n"
            "(a) Determine the serum IgE concentration\n"
            "(b) Determine the total eosinophil count\n"
            "(c) Refer her to an allergist\n"
            "(d) Treat the symptoms\n"
        ),
        "answer": "d",
    },
    # conceptual_physics
    {
        "prompt": (
            "If you were to travel at a speed close to the speed of light, you "
            "could notice that your own\n"
            "Options:\n"
            "(a) mass changes.\n"
            "(b) pulse decreases.\n"
            "(c) Both of these.\n"
            "(d) Neither of these.\n"
        ),
        "answer": "d",
    },
    # professional_psychology
    {
        "prompt": (
            "Form A of a standardized personality test was given in the fall "
            "and again in the spring co the same group of people. The "
            "reliability estimate that resulted from this research is referred "
            "to as\n"
            "Options:\n"
            "(a) external consistency\n"
            "(b) equivalence\n"
            "(c) stability\n"
            "(d) internal consistency\n"
        ),
        "answer": "c",
    },
    # high_school_biology
    {
        "prompt": (
            "All of the following statements are true EXCEPT\n"
            "Options:\n"
            "(a) thyroxine increases the rate of metabolism\n"
            "(b) insulin decreases storage of glycogen\n"
            "(c) vasopressin stimulates water reabsorption in the kidney\n"
            "(d) epinephrine increases blood sugar levels and heart rate\n"
        ),
        "answer": "b",
    },
    # high_school_geography
    {
        "prompt": (
            "The majority of Kurds are found in which country?\n"
            "Options:\n"
            "(a) Iran\n"
            "(b) Iraq\n"
            "(c) Turkey\n"
            "(d) Egypt\n"
        ),
        "answer": "c",
    },
    # miscellaneous
    {
        "prompt": (
            "Including the bottom how many sides are on a square-based "
            "pyramid?\n"
            "Options:\n"
            "(a) three\n"
            "(b) four\n"
            "(c) five\n"
            "(d) six\n"
        ),
        "answer": "c",
    },
    # electrical_engineering
    {
        "prompt": (
            "One of the following is the primary function of an oscillator\n"
            "Options:\n"
            "(a) produces sinusoidal oscillations\n"
            "(b) generates non sinusoidal waveforms\n"
            "(c) generates sustained oscillations at a constant amplitude and specific frequency\n"
            "(d) none of the above\n"
        ),
        "answer": "c",
    },
    # security_studies
    {
        "prompt": (
            "In what ways, if any, can the environment be considered a "
            "security concern?\n"
            "Options:\n"
            "(a) Environmental security entails a consideration of the security of the global environment, as well as its nested sub-systems and social systems beyond the boundaries of the nation state.\n"
            "(b) Environmental security is a critical security project in that it questions who and what is to be secured and from what threat by orthodox security policies, or whether linkages between environmental, security and development issues can be made.\n"
            "(c) Environmental security is a practical endeavour to assess how environmental change causes violent conflict within and between countries, and the ways in which environmental security can undermine national security.\n"
            "(d) All of these options. The environment is both an object to be secured and a source of risk, although it may mean different things to different people. Whilst deepening the concept of security it has both critical and practical dimensions although the utility of the concept is contested.\n"
        ),
        "answer": "d",
    },
    # world_religions
    {
        "prompt": (
            "What is the communal meal offered at the place of worship called "
            "in Sikhism?\n"
            "Options:\n"
            "(a) Sangat\n"
            "(b) Langar\n"
            "(c) Gurdwara\n"
            "(d) Panth\n"
        ),
        "answer": "b",
    },
    # sociology
    {
        "prompt": (
            "When Berger & Luckmann said that reality is socially constructed, "
            "they meant:\n"
            "Options:\n"
            "(a) scientists are guided in their work by social values and interests, so they define and measure phenomena that will support their theories\n"
            "(b) people negotiate shared definitions of their situation and live according to these, often forgetting that these social worlds are not fixed and external\n"
            "(c) sociologists decide what constitutes social reality and measure only that\n"
            "(d) terms like 'reality' have no deeper meaning beyond the level of discourse\n"
        ),
        "answer": "b",
    },
    # nutrition
    {
        "prompt": (
            "These factors increase risk of osteoporotic fracture:\n"
            "Options:\n"
            "(a) High bone mineral density\n"
            "(b) High body weight\n"
            "(c) High lean mass\n"
            "(d) Poor muscle strength\n"
        ),
        "answer": "d",
    },
    # college_biology
    {
        "prompt": (
            "To prevent desiccation and injury, the embryos of terrestrial "
            "vertebrates are encased within a fluid secreted by the\n"
            "Options:\n"
            "(a) amnion\n"
            "(b) chorion\n"
            "(c) allantois\n"
            "(d) yolk sac\n"
        ),
        "answer": "a",
    },
    # professional_psychology
    {
        "prompt": (
            "Identify the only construct that is not pertinent to "
            "developmental models on intelligence:\n"
            "Options:\n"
            "(a) Investment theory\n"
            "(b) The positive manifold\n"
            "(c) G theory\n"
            "(d) Primary mental ability theory\n"
        ),
        "answer": "a",
    },
    # professional_accounting
    {
        "prompt": (
            "Order the following (risk, return) pairs from least to most "
            "favourable, assuming the perspective of a rational and risk- "
            "averse investor: (2,2),(2,3) and (4,2). Risk is measured in "
            "standard deviations and return in percentage. HINT: Imagine a "
            "scatter diagram with standard deviation on the x-axis and return "
            "on the y-axis.\n"
            "Options:\n"
            "(a) (4,2),(2,2),(2,3)\n"
            "(b) (2,2),(2,3),(4,2)\n"
            "(c) (2,2),(4,2),(2,3)\n"
            "(d) (2,3),(2,2),(4,2)\n"
        ),
        "answer": "a",
    },
    # international_law
    {
        "prompt": (
            "What is personal (ratione personae) immunity?\n"
            "Options:\n"
            "(a) Personal immunity is afforded to all physical persons\n"
            "(b) Personal immunity is that which is afforded in a personal capacity and hence does not cover conduct of the State as such\n"
            "(c) Personal immunity is afforded only to particular persons irrespective if their conduct was undertaken in a private or public capacity\n"
            "(d) Personal immunity is afforded to State officials for conduct undertaken in a public capacity\n"
        ),
        "answer": "c",
    },
    # international_law
    {
        "prompt": (
            "Who is an 'injured State' in the law of international "
            "responsibility?\n"
            "Options:\n"
            "(a) A State is 'injured' in case that it has suffered a damage from the internationally wrongful conduct\n"
            "(b) A State is 'injured' in cases that there has been a violation of a peremptory norm of international law\n"
            "(c) A State is 'injured' should it acknowledge the existence of the internationally wrongful conduct\n"
            "(d) A State is 'injured' if the obligation breached was owed to it individually or if it was owed to a group of States, including that State, and it was specially affected\n"
        ),
        "answer": "d",
    },
    # marketing
    {
        "prompt": (
            "There are three main types of buying situations in an "
            "organization, referred to by Robinson, Faris, and Wind (1967) as "
            "_____________.\n"
            "Options:\n"
            "(a) Repeat purchases.\n"
            "(b) Buyphases.\n"
            "(c) Buyclasses.\n"
            "(d) Tenders.\n"
        ),
        "answer": "c",
    },
    # public_relations
    {
        "prompt": (
            "A ________ campaign occurs when people from two or more opposing "
            "sides of an argument have emotional convictions about a decision "
            "that has the power to impact their lives.\n"
            "Options:\n"
            "(a) public relations\n"
            "(b) public issues\n"
            "(c) crisis management\n"
            "(d) consumer relations\n"
        ),
        "answer": "b",
    },
    # high_school_psychology
    {
        "prompt": (
            "Which of the following samples would be considered most "
            "representative of male college students?\n"
            "Options:\n"
            "(a) A group of thirty fraternity brothers from Penn State\n"
            "(b) A random sample taken between classes in the business wing of various universities\n"
            "(c) Sixty male members of each class from Princeton, Yale, Harvard, Dartmouth, and Columbia\n"
            "(d) Twenty male members of each class from a cross-section of colleges and universities\n"
        ),
        "answer": "d",
    },
    # medical_genetics
    {
        "prompt": (
            "X-chromosome inactivation\n"
            "Options:\n"
            "(a) results in genetically turning off one of the two X chromosomes in female mammals\n"
            "(b) takes place in humans so that the same X chromosome is inactive in all of the cells of a female\n"
            "(c) is the cause of the Y chromosome being genetically inactive\n"
            "(d) occurs in fruit flies but not in mammals\n"
        ),
        "answer": "a",
    },
    # anatomy
    {
        "prompt": (
            "In the fetus, the ductus arteriosus passes blood from the\n"
            "Options:\n"
            "(a) pulmonary vein to the aorta.\n"
            "(b) aorta to pulmonary vein.\n"
            "(c) pulmonary artery to the aorta.\n"
            "(d) aorta to the pulmonary artery.\n"
        ),
        "answer": "c",
    },
    # human_sexuality
    {
        "prompt": (
            "The follicular phase is to the __________ as the luteal phase is "
            "to the secretory phase.\n"
            "Options:\n"
            "(a) postovulatory\n"
            "(b) menstrual\n"
            "(c) proliferative\n"
            "(d) myometrial\n"
        ),
        "answer": "c",
    },
    # computer_security
    {
        "prompt": (
            "Buffer-overflow may remain as a bug in apps if __________ are not "
            "done fully.\n"
            "Options:\n"
            "(a) boundary hacks\n"
            "(b) memory checks\n"
            "(c) boundary checks\n"
            "(d) buffer checks\n"
        ),
        "answer": "c",
    },
    # philosophy
    {
        "prompt": (
            "According to Socrates, it is important that we discover what "
            "makes a particular action (e.g., a merciful or just act) the kind "
            "of action that it is, because without such knowledge:\n"
            "Options:\n"
            "(a) no one in society will ever do any action that really is merciful or just, only those actions that they think are merciful or just.\n"
            "(b) the primary purpose of human existence--which is to think and to know--is replaced by a focus on morality (acting and doing).\n"
            "(c) we can refer only to how people characterize actions without knowing why such actions should be characterized that way.\n"
            "(d) there would be no way to distinguish one kind of action (e.g., a merciful action) from another kind of action (e.g., a just action).\n"
        ),
        "answer": "c",
    },
    # high_school_macroeconomics
    {
        "prompt": (
            "What does the presence of discouraged workers do to the "
            "measurement of the unemployment rate?\n"
            "Options:\n"
            '(a) Discouraged workers are counted as "out of the labor force" thus the unemployment rate is understated making the economy look stronger than it is.\n'
            '(b) Discouraged workers are counted as "out of the labor force" thus the unemployment rate is understated making the economy look weaker than it is.\n'
            "(c) Discouraged workers are not surveyed so there is no impact on the unemployment rate.\n"
            '(d) Discouraged workers are counted as "unemployed" thus the unemployment rate is understated making the economy look stronger than it is.\n'
        ),
        "answer": "a",
    },
    # miscellaneous
    {
        "prompt": (
            "In the film 'The Talented Mr Ripley' who plays Mr Ripley?\n"
            "Options:\n"
            "(a) Jude Law\n"
            "(b) Matt Damon\n"
            "(c) Dustin Hoffman\n"
            "(d) Ben Affleck\n"
        ),
        "answer": "b",
    },
    # anatomy
    {
        "prompt": (
            "Which of the following allows gas exchange in the lungs?\n"
            "Options:\n"
            "(a) Alveoli\n"
            "(b) Bronchi\n"
            "(c) Bronchioles\n"
            "(d) Capillaries\n"
        ),
        "answer": "a",
    },
    # philosophy
    {
        "prompt": (
            "Anscombe criticizes as absurd Kant's idea of:\n"
            "Options:\n"
            "(a) the thing in itself.\n"
            "(b) the categorical imperative.\n"
            "(c) the phenomenal self.\n"
            "(d) legislating for oneself.\n"
        ),
        "answer": "d",
    },
    # professional_medicine
    {
        "prompt": (
            "During a clinical study examining the effects of exercise, men "
            "between the ages of 20 and 30 years are evaluated during a 15- "
            "minute session on a treadmill. The average pulse for the last 2 "
            "minutes of the session is 175/min. During the last minute of "
            "exercise, various measurements are taken. Compared with the "
            "measurement before the session, which of the following is most "
            "likely to be decreased?\n"
            "Options:\n"
            "(a) Pulse pressure\n"
            "(b) Stroke volume\n"
            "(c) Systolic blood pressure\n"
            "(d) Total peripheral resistance\n"
        ),
        "answer": "d",
    },
    # prehistory
    {
        "prompt": (
            "Which of the following was described as the slow agency of "
            "existing causes?\n"
            "Options:\n"
            "(a) natural selection\n"
            "(b) catastrophism\n"
            "(c) uniformitarianism\n"
            "(d) unilineal evolution\n"
        ),
        "answer": "c",
    },
    # high_school_psychology
    {
        "prompt": (
            "A school psychologist is providing feedback to a student's "
            "parents regarding the student's performance on a measure of "
            "academic achievement. To explain the concept of grade equivalent, "
            "the school psychologist should explain that it is\n"
            "Options:\n"
            "(a) the average score on that measure obtained by students in a given grade\n"
            "(b) the average score on that measure obtained by students at a given age\n"
            "(c) the grade in which a student should be placed in school\n"
            "(d) utilized to determine accountability among peers\n"
        ),
        "answer": "a",
    },
    # us_foreign_policy
    {
        "prompt": (
            "Global and regional international trade agreements work by using "
            "which of the following mechanisms?\n"
            "Options:\n"
            "(a) Reciprocity across multiple issues\n"
            "(b) Reputational concerns of the actors\n"
            "(c) Side payments for adjusting to the organization (such as the Common Agricultural Policy in the EU)\n"
            "(d) ALL of the above\n"
        ),
        "answer": "d",
    },
    # prehistory
    {
        "prompt": (
            "When did Mohenjo-daro develop into a complex urban center?\n"
            "Options:\n"
            "(a) during the Early Shang Dynasty, from 1,000 to 400 B.P.\n"
            "(b) during the Qin Dynasty, from 3,500 to 3,000 B.P.\n"
            "(c) during the Mature Harappan period, from 4,500 to 4,000 B.P.\n"
            "(d) during the Early Harappan period, from 8,500 to 6,000 B.P.\n"
        ),
        "answer": "c",
    },
    # nutrition
    {
        "prompt": (
            "Adaptive thermogenesis refers to:\n"
            "Options:\n"
            "(a) A decrease in heat loss when exposed to cold\n"
            "(b) A decrease in non-shivering thermogenesis when exposed to cold\n"
            "(c) An increase in basal metabolic rate that is not fully explained by a change in body composition during chronic overfeeding\n"
            "(d) A decrease in spontaneous physical activity during chronic overfeeding.\n"
        ),
        "answer": "c",
    },
    # human_sexuality
    {
        "prompt": (
            "In all of her relationships Helen is typically uncomfortable "
            "feeling close to another person or having that person feel close "
            "to her. According to attachment theory Helen would best be "
            "described as:\n"
            "Options:\n"
            "(a) a secure lover\n"
            "(b) an avoidant lover\n"
            "(c) an anxious-ambivalent lover\n"
            "(d) a passionate lover\n"
        ),
        "answer": "b",
    },
    # prehistory
    {
        "prompt": (
            "In primates, the location of which of the following determines if "
            "it is a quadruped or biped?\n"
            "Options:\n"
            "(a) the feet\n"
            "(b) the skull\n"
            "(c) the foramen magnum\n"
            "(d) the ulna\n"
        ),
        "answer": "c",
    },
    # international_law
    {
        "prompt": (
            "What is the meaning of collective security?\n"
            "Options:\n"
            "(a) The right to self-defence by more than one nation acting in concert\n"
            "(b) The right of one's allies to defend the victim State\n"
            "(c) The authorisation of armed force by the UN Security Council\n"
            "(d) The authorisation of peacekeeping missions by the UN General Assembly\n"
        ),
        "answer": "c",
    },
    # miscellaneous
    {
        "prompt": (
            "In the comic strip 'Peanuts' what is Schroeder known for doing?\n"
            "Options:\n"
            "(a) Dancing\n"
            "(b) playing football\n"
            "(c) playing the piano\n"
            "(d) Flying an imaginary plane\n"
        ),
        "answer": "c",
    },
    # college_biology
    {
        "prompt": (
            "Which of the following is the symplastic pathway for the movement "
            "of sucrose from the site of photosynthesis in mesophyll cells "
            "into the phloem?\n"
            "Options:\n"
            "(a) Fibers, phloem parenchyma, companion cell, sieve tube\n"
            "(b) Phloem parenchyma, fibers, bundle sheath, tracheids\n"
            "(c) Companion cells, phloem parenchyma, fibers, sieve tube\n"
            "(d) Bundle sheath, phloem parenchyma, companion cell, sieve tube\n"
        ),
        "answer": "d",
    },
    # professional_medicine
    {
        "prompt": (
            "A 45-year-old man comes to the physician because of right "
            "shoulder pain that began after he chopped wood 2 days ago. "
            "Examination of the right upper extremity shows no obvious bone "
            "deformities or point tenderness. The pain is reproduced when the "
            "patient is asked to externally rotate the shoulder against "
            "resistance; there is no weakness. In addition to the teres minor, "
            "inflammation of which of the following tendons is most likely in "
            "this patient?\n"
            "Options:\n"
            "(a) Infraspinatus\n"
            "(b) Pectoralis\n"
            "(c) Subscapularis\n"
            "(d) Supraspinatus\n"
        ),
        "answer": "a",
    }
]


# ─── Dataset loaders ───────────────────────────────────────────────

def load_bbh_questions(subtask, limit=50):
    ds = load_dataset("SaylorTwift/bbh", subtask)
    test = ds["test"]
    total = len(test)
    step = max(1, total // limit)
    indices = list(range(0, total, step))[:limit]

    questions = []
    for idx in indices:
        item = test[idx]
        answer_match = re.search(r'the answer is (.+?)\.?\s*$', item["target"], re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            answer = item["target"].strip().split()[-1].rstrip(".")

        questions.append({
            "prompt": item["input"] + "\n\nThink step by step, then give your final answer.",
            "answer": answer,
            "type": f"bbh_{subtask}",
        })
    return questions


def load_tinymmlu_questions():
    questions = []
    for mmlu_question in TINYMMLU_QUESTIONS:
        questions.append({
            "prompt": mmlu_question["prompt"] + "Give your answer as a single letter only.",
            "answer": mmlu_question["answer"],
            "type": "tinymmlu"
        })
    return questions


# ─── Scoring ───────────────────────────────────────────────────────

def extract_bbh_final_answer(response: str) -> str:
    match = re.search(r'the answer is\s+(.+?)[\.\!\n]', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r'####\s*(.+)', response)
    if match:
        return match.group(1).strip()
    lines = response.strip().split('\n')
    return lines[-1].strip()


def extract_mmlu_final_answer(response: str) -> str:
    match = re.search(r'answer is \([a-dA-D]\)', response)
    if match:
        return match.group(0)[11:12].lower()
    match = re.search(r'\([a-dA-D]\)', response)
    if match:
        return match.group(0)[1:2].lower()
    match = re.fullmatch(r'[a-dA-D]\s*', response)
    if match:
        return match.group(0)[0:1].lower()
    return "x"


def score_question(question: dict, response: str | None) -> int:
    if response is None:
        return {"score": 0, "parsed": None, "correct": question["answer"]}

    if question["type"] != "tinymmlu":
        final = extract_bbh_final_answer(response)
        correct = question["answer"].strip().lower()
        final_clean = final.strip().lower()
        final_clean = re.sub(r'[^a-z0-9\s\(\)]', '', final_clean).strip()
        correct_clean = re.sub(r'[^a-z0-9\s\(\)]', '', correct).strip()
    else:
        final = extract_mmlu_final_answer(response)
        final_clean = final
        correct_clean = question["answer"]

    if correct_clean in final_clean or final_clean == correct_clean:
        return 1
    elif correct_clean in ("yes", "no") and correct_clean in final_clean.split():
        return 1
    return 0


# ─── Main evaluation ───────────────────────────────────────────────

def run_full_evaluation(model: str, port: int, bbh_limit: int = 50,
                        max_tokens: int = 350) -> dict:
    print("Loading questions...")
    all_questions = []

    for subtask in ["causal_judgement", "date_understanding",
                     "logical_deduction_five_objects", "navigate",
                     "boolean_expressions", "tracking_shuffled_objects_three_objects"]:
        try:
            bbh = load_bbh_questions(subtask, limit=bbh_limit)
            all_questions.extend(bbh)
            print(f"  BBH {subtask}: {len(bbh)} questions")
        except Exception as e:
            print(f"  BBH {subtask}: FAILED ({e})")

    tinymmlu = load_tinymmlu_questions()
    all_questions.extend(tinymmlu)
    print(f"  TinyMMLU: {len(tinymmlu)} questions")

    total_q = len(all_questions)
    print(f"\nTotal: {total_q} questions")
    print("Running evaluation...\n")

    results_by_type = {}
    start_time = time.time()

    for i, q in enumerate(all_questions):
        qtype = q["type"]
        if qtype not in results_by_type:
            results_by_type[qtype] = {"total": 0, "correct": 0}

        response = query_model(q["prompt"], port, max_tokens)
        score = score_question(q, response)

        results_by_type[qtype]["total"] += 1
        results_by_type[qtype]["correct"] += score

        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (total_q - i - 1) / rate if rate > 0 else 0
        print(f"\r  [{i+1}/{total_q}] {qtype:50s} "
              f"score={score}  "
              f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", end="", flush=True)

    print("\n")
    total_elapsed = time.time() - start_time

    print("=" * 70)
    print(f"{'Probe':50s} {'Accuracy':>8} {'N':>5}")
    print("-" * 70)

    summary = {}

    for qtype in sorted(results_by_type.keys()):
        data = results_by_type[qtype]
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        print(f"  {qtype:48s} {acc:>8.1%} {data['total']:>5}")
        summary[qtype] = {"accuracy": acc, "n": data["total"]}

    correct_q = sum(r["correct"] for r in results_by_type.values())
    overall_avg = correct_q / total_q if total_q > 0 else 0
    print("-" * 70)
    print(f"  {'OVERALL':48s} {overall_avg:>8.1%} {total_q:>5}")
    print(f"\nCompleted in {total_elapsed:.0f}s")
    print("=" * 70)

    return {
        "model": model,
        "summary": summary,
        "overall": overall_avg,
        "elapsed": total_elapsed,
        "n_questions": total_q,
    }


def main():
    parser = argparse.ArgumentParser(description="Additional RYS evaluation")
    parser.add_argument("--model", required=True, help="Path to input GGUF model")
    parser.add_argument("--llama-server", required=True, help="Path to llama-server binary")
    parser.add_argument("--tmpdir", default="/dev/shm/rys",
                        help="Temp directory for modified GGUFs (use tmpfs/RAM)")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--bbh-limit", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--server-args", nargs=argparse.REMAINDER, default=[],
                        help="Extra args to pass to llama-server (must be last)")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    tmpdir = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    print("\n>>> Running evaluation...")
    proc = start_server(args.llama_server, str(model_path), tmpdir, args.port, args.server_args)
    try:
        if not wait_for_server(args.port):
            print("ERROR: Server failed to start for baseline", file=sys.stderr)
            dump_server_log(proc)
            stop_server(proc)
            sys.exit(1)

        print("  Server ready. Running probes...")
        results = run_full_evaluation(args.model, args.port, args.bbh_limit, args.max_tokens)
    finally:
        stop_server(proc)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
