# Implementation of a Contextual Chatbot in PyTorch.  
Simple chatbot implementation with PyTorch. 
this chatbot was built to provide guest or foreigner about all the informations about needed about our school

## Installation

### Create an environment
Whatever you prefer (e.g. `conda` or `venv`)
```console
mkdir myproject
$ cd myproject
$ python3 -m venv venv
```

### Activate it
Mac / Linux:
```console
. venv/bin/activate
```
Windows:
```console
venv\Scripts\activate
```
### Install PyTorch and dependencies

For Installation of PyTorch see [official website](https://pytorch.org/).

You also need `nltk`:
 ```console
pip install nltk
 ```

If you get an error during the first run, you also need to install `nltk.tokenize.punkt`:
Run this once in your terminal:
 ```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage
Run
```console
python train.py
```
This will dump `data.pth` file. And then run
```console
python chat.py
```
## Customize
Have a look at [intents.json](intents.json). You can customize it according to your own use case. Just define a new `tag`, possible `patterns`, and possible `responses` for the chat bot. You have to re-run the training whenever this file is modified.
```console
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "hi",
        "Hey",
        "hello",
        "How are you?",
        "is anyone there?",
        "Good evening",
        "Good morning"
      ],
      "responses": [
        "Hey :-)",
        "Hello, thanks for visiting",
        "Good to see you again",
        "Hi there, how can I help?"
      ]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye","Goodbye", "See you later"],
      "responses": [
        "See you later, thanks for visiting",
        "have a nice day",
        "Bye! Come back again soon."
      ]
    },
    {
      "tag": "help",
      "patterns": ["What can you do?","What are your features?","What are you abilities?","i need help","help me","who are you?","What is your name?","how could you help me"],
      "responses": ["hi there, i'm ensa berrechid bot i can give you all the information about ensa berrechid"]
    },
    {"tag": "thanks",
         "patterns": ["Thanks", "Thank you", "That's helpful","nice one"],
         "responses": ["Happy to help!", "Any time!", "My pleasure"]
        },
    {"tag": "sure","patterns": ["are you sure?","are you certain?"],
    "responses": ["yes,of course"]},
    {"tag": "creator",
    "patterns": ["Who is your creator?", "who created you?", "who is your father?","who is your daddy?"],
    "responses": ["i was created by Mr.saidi and Mr.Elbaidoury"]},
    {
      "tag": "contact",
      "patterns": [
        "Where is the school located?",
        "location",
        "school address",
        "I want to contact the school",
        "contact",
        "how can i contact you?",
        "I'm trying to contact you",
        "mail address",
        "school phone number","Email",
        "phone","Fax"
      ],
      "responses": [
        "Adresse: Avenue de l'universite, B.P :218 Berrechid.\n phone: 05-22-32-47-58 \n Fax: 05-22-53-45-30 \n Email: ensa.berrechid@uhp.ac.ma"
      ]
    },
    {"tag": "dut","patterns": ["what dut means?","can you explain dut?","dut"],
    "responses": ["The University Diploma of Technology (DUT) is a national diploma of Moroccan higher education which is prepared in two years \n it is oriented towards the professional integration of students,but also offers a solid \n theoretical training which allows the pursuit of studies.\n in professional license, engineering school, Moroccan and foreign faculties."]},
    {
      "tag": "cycledut",
      "patterns": ["what are the specialties of the dut cycle?","how many speciality there is in the dut cycle?"],
      "responses": ["there is four specialties in the dut cycle: \n-Electrical Engineering.\n-Computer Engineering.\n-Logistics and Transport Engineering.\nManagement techniques"]
    },
    {"tag": "cyclelisense",
    "patterns": ["what are the specialties of the professional license?","how many speciality there is in the professional license?"],
    "responses":["there is three specialties in the professional license:\n-Electrical Engineering and Renewable Energies.\n-Management and business administration.\n-IT and Engineering of Decision-Making Systems"]},
    {"tag": "cycleing",
    "patterns": ["what are the specialties of the cycle of engineer?","how many speciality there is in the cycle of engineer?"],
    "responses": ["there is two specialties in the cycle of engineer:\n-Aeronautical engineering.\n-Information Systems and Big Data Engineering."]},
    {
      "tag": "introduction",
      "patterns": ["give me an introduction about the school","give me an introduction about ensa berrechid","what is ensa berrechid?","school history","ensa berrechid history"],
      "responses": ["National School of Applied Sciences of Berrechid (ENSA Berrechid) is a public engineering school \n under the Hassan 1er University of Settat and part of the ENSA network of Morocco. ENSA Berrechid has for \n vocation to train engineers and specialists in scientific and technical disciplines \n diversified. \n By a ministerial decree of May 11, 2018, EST Berrechid has been transformed into ENSA Berrechid since \n 2018/2019 university start, while keeping the training courses previously provided as part of the \n DUT and Professional License \n"]
    },
    {"tag": "defaero","patterns": ["Aeronautical engineering","Aerospace","definition about Aeronautical engineering"],
    "responses": ["Aeronautical engineering is the design, production, testing and maintenance of aircraft, aerospace vehicles and their systems. This includes conventional fixed-wing aircraft as well as helicopters, spacecraft and drones."]},
    {"tag": "defbigdata","patterns": ["Big data","definition about big data","what big data mean?"],
    "responses": ["Big data is a field that treats ways to analyze, systematically extract information from, or otherwise deal with data sets that are too large or complex to be dealt with by traditional data-processing application software"]
    },
    {
      "tag": "diplome",
      "patterns": ["Diplomas","what do you present as diplomas?","what are the diplomas delivered by ensa berrechid?","can i know the diplomas issued by your school?"],
      "responses":["Ensa berrechid delivere three type of diplomas:\n-State Engineer diploma.\n-professional license.\n-technical University degree(DUT)."]
    },
    {
      "tag": "directeur",
      "patterns": ["director","who is in charge of the school?","who is the principal of the school?"],
      "responses":["Mr. ABDELMOUMEN TABYAOUI here is his LinkedIn account :\n https://www.linkedin.com/in/tabyaoui-abdelmoumen-159284a0/?originalSubdomain=ma"]
    },
    {"tag": "accesscycleing",
    "patterns": ["how can i access to the cycle of engineer?","acesss to the cycle of engineer","how can i access to ensa berrechid","how can i access to you school","what are the access condition to the cycle of engineer?"],
    "responses": ["there is the steps that you should follow to access the cycle of engineer:\n-1)Connect to the platform by entering your CIN number or Passport number and your password.\n-2)Choose one of the courses below and click on pre-registration.\n-3)Choose one of the courses below and click on pre-registration.\n-4)Download your pre-registration receipt.\n-5)follow the display of the lists of shortlisted candidates to take the competition.\n-6)written competition according to the calendar"]},
    {"tag": "presentationcycleing",
    "patterns": ["a presentation about the cycle of engineer","definition about the cycle of engineer","what the cycle of engineer means?"],
    "responses": ["The ENSA Berrechid curriculum begins with the first two years common to all students.These two years are managed by the preparatory classes for the engineering cycle.The purpose of this preparatory cycle is to move students from the status of high school student to that of future engineer and their provide basic training that allows them to continue their studies, regardless of the course chosen in the engineering cycle."]},
    {"tag": "modules",
    "patterns": ["what are the modules taught in the engineering cycle?","what are the modules teached in the engineering cycle?","what are the modules teached in Aeronautical engineering?","what are the modules taught in the information systems and big data engineering"],
    "responses": ["for more detailed answers visite the link bellow:\n-http://www.ensab.ac.ma/Emploi"]},
    {
      "tag": "funny",
      "patterns": [
        "tell me a joke?",
        "make me laugh",
        "tell me a science joke",
        "tell me something funny"
      ],
      "responses": [
        "Why did the hipster burn his mouth? He drank the coffee before it was cool.",
        "What did the buffalo say when his son left for college? Bison.",
        "How do you make holy water? you boil the hell out of it",
        "Did you hear oxygen went on a date with potassium? A: It went OK."
      ]
    }
  ]
}

```
