{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#for feature engineering\n",
    "import pandas as pd \n",
    "import numpy as np\n",
    "\n",
    "from sklearn.naive_bayes import ComplementNB # specifically for handeling imbalabced data\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from skmultilearn.adapt import MLkNN\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.decomposition import TruncatedSVD\n",
    "\n",
    "from scipy.sparse import csr_matrix\n",
    "from scipy.sparse import csr_matrix, lil_matrix\n",
    "\n",
    "\n",
    "# NLP\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.feature_extraction.text import TfidfTransformer\n",
    "\n",
    "# Metrics\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.metrics import roc_auc_score\n",
    "\n",
    "# Wrappers\n",
    "from sklearn.multiclass import OneVsRestClassifier # run model column by colum, treated as seperated models\n",
    "from sklearn.multioutput import MultiOutputClassifier #\n",
    "from sklearn.multioutput import ClassifierChain # the next model's input is its previous models output+feature\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "class NB_models:\n",
    "    \n",
    "    '''this class allows you to specify classifiers which have to has predict_proba method. \n",
    "    To handle multi-lable problem, the model used two wrappers: OneVsRest and ClassifierChain, or it will use mlKnn.\n",
    "    roc_aus will be used as model evaluation metrics\n",
    "    ----------------------------------------\n",
    "    \n",
    "    Parameters: \n",
    "    \n",
    "    To initialize a class object, you need to specify X_train, X_train, X_test:\n",
    "        X_train: train features from clear csv with our engineered features. note: NB models works best on count numbers, and doesn't work if you \n",
    "        have negative values\n",
    "        \n",
    "        X_train: lables. note: because we used ClassifierChain, the order of lables will influence on \n",
    "        the prediction results  \n",
    "        \n",
    "        X_test: tested features from clear csv with our engineered features.\n",
    "        \n",
    "    To use model_base method, you need to input:\n",
    "        classifier: a classifier with predict_proba method \n",
    "        \n",
    "        proba_threshold: with default = 0.5 \n",
    "        \n",
    "        use_transformer: with default = False, you can specify CountVectorizer() or TfidfTransformer() \n",
    "        \n",
    "        tag_transformer： with defautl = False, you can use True to let the transformed LSA data tag along with the featured data\n",
    "    ---------------------------------------- \n",
    "    Models that you can test using this class:\n",
    "      -  basemodel 1: modeling with only feature engineering columns\n",
    "      -  basemodel 2: modeling with only countvectorizer on the clean text columns\n",
    "      -  model 3 - FE+CV model: model 1 + model 2: used LSA to reduce dimensionality for countervectorized data and tag this data to other featured\n",
    "         data, however after LSA,  NB models are not spplicable since it works best on calcaulation probabilities directly. \n",
    "         So used random forest instead.\n",
    "    '''\n",
    "    \n",
    "    \n",
    "    def __init__(self, X_train, y_train, X_test):\n",
    "        self.X_train = X_train\n",
    "        self.y_train = y_train\n",
    "        self.X_test = X_test\n",
    "    \n",
    "    def model_base(self, classifier, proba_threshold =0.5, use_transformer = False, tag_transformer = False):\n",
    "        '''The classifier name has to be consist with SKlearn'''\n",
    "        self.proba_threshold = proba_threshold\n",
    "        '''Any classfier is fine, but please pay attention on any prerequisite of data that are needed to \n",
    "        feed in the model'''\n",
    "        self.model = classifier\n",
    "        '''CountVectorizer or TfidfTransformer'''\n",
    "        self.use_transformer = use_transformer\n",
    "        \n",
    "        self.tag_transformer = tag_transformer\n",
    "        \n",
    "        roc_aucs = []\n",
    "        estimators = []\n",
    "        res = {}\n",
    "        self.X_train, self.X_test = self.dataprep()\n",
    "        print(self.X_train.shape,self.y_train.shape,self.X_test.shape)\n",
    "        \n",
    "        ##### iter through different strategies\n",
    "        strategies = [OneVsRestClassifier, ClassifierChain]\n",
    "        for i in strategies:\n",
    "\n",
    "            clf = i(self.model) \n",
    "            clf.fit(self.X_train, self.y_train)\n",
    "\n",
    "            y_pred = pd.DataFrame(clf.predict_proba(self.X_test))\n",
    "           \n",
    "            y_pred = y_pred.applymap(lambda x: 1 if x> self.proba_threshold else 0)\n",
    "\n",
    "            #####for evaluation\n",
    "            y_train_hat = clf.predict(self.X_train)\n",
    "\n",
    "            ##### evaluation is on train data\n",
    "            roc_aucs = roc_auc_score(self.y_train, y_train_hat, average = None)\n",
    "            roc_aucs_weighted = roc_auc_score(self.y_train, y_train_hat, average = 'weighted')\n",
    "\n",
    "         ##### Store results into dictionary   \n",
    "            res[i] = (roc_aucs,roc_aucs_weighted,y_pred)\n",
    "\n",
    "        return (res)   \n",
    "\n",
    "    def mlknn_model(self, proba_threshold =0.5, k=10, s=1, use_transformer = False, tag_transformer = False):\n",
    "        '''CountVectorizer or TfidfTransformer'''\n",
    "        self.use_transformer = use_transformer\n",
    "        self.tag_transformer = tag_transformer  \n",
    "        self.X_train, self.X_test = self.dataprep()\n",
    "        \n",
    "        '''The classifier name has to be consist with SKlearn'''\n",
    "        print(self.X_train.shape,self.y_train.shape,self.X_test.shape)\n",
    "    \n",
    "        self.proba_threshold = proba_threshold\n",
    "        roc_aucs = []\n",
    "        estimators = []\n",
    "        res = {}\n",
    "      \n",
    "        clf = MLkNN(k, s)\n",
    "        model = clf.fit(lil_matrix(self.X_train), lil_matrix(self.y_train))\n",
    "     \n",
    "        y_pred = pd.DataFrame(clf.predict_proba(self.X_test).toarray())\n",
    "        y_pred = y_pred.applymap(lambda x: 1 if x > self.proba_threshold else 0)\n",
    "   \n",
    "        #####for evaluation\n",
    "        y_train_hat = clf.predict(self.X_train)\n",
    "\n",
    "        ##### evaluation is on train data\n",
    "        roc_aucs = roc_auc_score(self.y_train, y_train_hat.todense(), average = None)\n",
    "        roc_aucs_weighted = roc_auc_score(self.y_train, y_train_hat.todense(), average = 'weighted')\n",
    "\n",
    "        ##### Store results \n",
    "        res = (roc_aucs,roc_aucs_weighted,y_pred)\n",
    "\n",
    "        return (res) \n",
    "    \n",
    "    \n",
    "    def dataprep(self):\n",
    "        '''if using any transformers'''\n",
    "        if (self.use_transformer != False and \n",
    "            self.tag_transformer == False):\n",
    "            print('This will only use transformed data, please check carefully')\n",
    "            \n",
    "            if 'TfidfTransformer' in str(self.use_transformer):\n",
    "                ct_vectorizer2 = CountVectorizer()\n",
    "                vectorized_data_train = ct_vectorizer2.fit_transform(self.X_train)\n",
    "             \n",
    "                self.X_train = self.use_transformer.fit_transform(vectorized_data_train)\n",
    "                vectorized_data_test = ct_vectorizer2.transform(self.X_test)\n",
    "                self.X_test =  self.use_transformer.transform(vectorized_data_test)\n",
    "\n",
    "            else:    \n",
    "                self.X_train = self.use_transformer.fit_transform(self.X_train)\n",
    "                self.X_test =  self.use_transformer.transform(self.X_test)\n",
    "        \n",
    "        '''if tagging along the LSA reduced matrix with created features'''\n",
    "        if (self.use_transformer != False \n",
    "            and self.tag_transformer != False):        \n",
    "            self.X_train_text = X_train['comment_text_clean']\n",
    "            self.X_train = X_train.drop('comment_text_clean', axis =1)\n",
    "            self.X_test_text = X_test['comment_text_clean']\n",
    "            self.X_test = X_test.drop('comment_text_clean', axis =1)\n",
    "            \n",
    "            print('This will only use LSA reduced matrix with created features, please check carefully')\n",
    "            if 'TfidfTransformer' in str(self.use_transformer):\n",
    "                ct_vectorizer = CountVectorizer()\n",
    "                vectorized_data_train = ct_vectorizer.fit_transform(self.X_train_text)\n",
    "                self.X_train_transformed = self.use_transformer.fit_transform(vectorized_data_train)\n",
    "                vectorized_data_test = ct_vectorizer.transform(self.X_test_text)\n",
    "                self.X_test_transformed =  self.use_transformer.transform(vectorized_data_test)\n",
    "\n",
    "            else:\n",
    "                self.X_train_transformed = self.use_transformer.fit_transform(self.X_train_text)\n",
    "                self.X_test_transformed =  self.use_transformer.transform(self.X_test_text)\n",
    "  \n",
    "        \n",
    "            svd = TruncatedSVD(5)\n",
    "            self.X_train_reduced_tfid = svd.fit_transform(self.X_train_transformed)#_text)\n",
    "            self.X_test_reduced_tfid = svd.transform(self.X_test_transformed)#_text)\n",
    "        \n",
    "            self.X_train_reduced_tfid = pd.DataFrame(self.X_train_reduced_tfid ,columns = [\"svd_\"+str(i) for i in list(range(self.X_train_reduced_tfid.shape[1]))])\n",
    "            self.X_test_reduced_tfid = pd.DataFrame(self.X_test_reduced_tfid ,columns = [\"svd_\"+str(i) for i in list(range(self.X_test_reduced_tfid.shape[1]))])\n",
    "            self.X_train = pd.concat([self.X_train, self.X_train_reduced_tfid], axis = 1, ignore_index = True)\n",
    "            self.X_test = pd.concat([self.X_test, self.X_test_reduced_tfid], axis = 1, ignore_index = True)\n",
    "\n",
    "        return (self.X_train, self.X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Load data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "train = pd.read_csv('../kaggle/train_clean.csv')[:3000]\n",
    "test  = pd.read_csv('../kaggle/test_clean.csv')[:3000]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>id</th>\n",
       "      <th>comment_text</th>\n",
       "      <th>toxic</th>\n",
       "      <th>severe_toxic</th>\n",
       "      <th>obscene</th>\n",
       "      <th>threat</th>\n",
       "      <th>insult</th>\n",
       "      <th>identity_hate</th>\n",
       "      <th>clean</th>\n",
       "      <th>...</th>\n",
       "      <th>comment_text_clean</th>\n",
       "      <th>num_website</th>\n",
       "      <th>num_rep_words</th>\n",
       "      <th>percentage_repeated_word</th>\n",
       "      <th>num_nouns</th>\n",
       "      <th>num_adjectives</th>\n",
       "      <th>num_verbs</th>\n",
       "      <th>percentage_nouns</th>\n",
       "      <th>percentage_adjs</th>\n",
       "      <th>percentage_verbs</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0000997932d777bf</td>\n",
       "      <td>Explanation\\nWhy the edits made under my usern...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>explanation why edit make username hardcore me...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>18</td>\n",
       "      <td>3</td>\n",
       "      <td>7</td>\n",
       "      <td>0.418605</td>\n",
       "      <td>0.069767</td>\n",
       "      <td>0.162791</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>000103f0d9cfb60f</td>\n",
       "      <td>D'aww! He matches this background colour I'm s...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>daww he match background colour i am seemingly...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>6</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>0.352941</td>\n",
       "      <td>0.117647</td>\n",
       "      <td>0.176471</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>000113f07ec002fd</td>\n",
       "      <td>Hey man, I'm really not trying to edit war. It...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>hey man i am really try edit war it is guy con...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>11</td>\n",
       "      <td>5</td>\n",
       "      <td>5</td>\n",
       "      <td>0.261905</td>\n",
       "      <td>0.119048</td>\n",
       "      <td>0.119048</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>0001b41b1c6bb37e</td>\n",
       "      <td>\"\\nMore\\nI can't make any real suggestions on ...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>more i cannot make real suggestions improveme...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>28</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>0.247788</td>\n",
       "      <td>0.097345</td>\n",
       "      <td>0.097345</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>0001d958c54c6e35</td>\n",
       "      <td>You, sir, are my hero. Any chance you remember...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>you sir hero any chance remember page that is on</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0.230769</td>\n",
       "      <td>0.076923</td>\n",
       "      <td>0.153846</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 67 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0                id  \\\n",
       "0           0  0000997932d777bf   \n",
       "1           1  000103f0d9cfb60f   \n",
       "2           2  000113f07ec002fd   \n",
       "3           3  0001b41b1c6bb37e   \n",
       "4           4  0001d958c54c6e35   \n",
       "\n",
       "                                        comment_text  toxic  severe_toxic  \\\n",
       "0  Explanation\\nWhy the edits made under my usern...      0             0   \n",
       "1  D'aww! He matches this background colour I'm s...      0             0   \n",
       "2  Hey man, I'm really not trying to edit war. It...      0             0   \n",
       "3  \"\\nMore\\nI can't make any real suggestions on ...      0             0   \n",
       "4  You, sir, are my hero. Any chance you remember...      0             0   \n",
       "\n",
       "   obscene  threat  insult  identity_hate  clean  ...  \\\n",
       "0        0       0       0              0      1  ...   \n",
       "1        0       0       0              0      1  ...   \n",
       "2        0       0       0              0      1  ...   \n",
       "3        0       0       0              0      1  ...   \n",
       "4        0       0       0              0      1  ...   \n",
       "\n",
       "                                  comment_text_clean  num_website  \\\n",
       "0  explanation why edit make username hardcore me...            0   \n",
       "1  daww he match background colour i am seemingly...            0   \n",
       "2  hey man i am really try edit war it is guy con...            0   \n",
       "3   more i cannot make real suggestions improveme...            0   \n",
       "4   you sir hero any chance remember page that is on            0   \n",
       "\n",
       "   num_rep_words  percentage_repeated_word  num_nouns  num_adjectives  \\\n",
       "0              0                       0.0         18               3   \n",
       "1              0                       0.0          6               2   \n",
       "2              0                       0.0         11               5   \n",
       "3              0                       0.0         28              11   \n",
       "4              0                       0.0          3               1   \n",
       "\n",
       "   num_verbs  percentage_nouns  percentage_adjs  percentage_verbs  \n",
       "0          7          0.418605         0.069767          0.162791  \n",
       "1          3          0.352941         0.117647          0.176471  \n",
       "2          5          0.261905         0.119048          0.119048  \n",
       "3         11          0.247788         0.097345          0.097345  \n",
       "4          2          0.230769         0.076923          0.153846  \n",
       "\n",
       "[5 rows x 67 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "train.drop('Unnamed: 0', axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2 TESTING BASE MODELS\n",
    "Models have to be complied with predict_proba funtion, otherwise this class will not work"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = train[['num_words', 'num_stopwords', 'percentage_non_stop', 'num_caps','num_toxic_symbol', \n",
    "    'num_happy', 'num_sad',  'num_unique_words', 'punctuation', \n",
    "       'exclamation_marks', 'exclamation_marks_vs_length', 'num_newlines',\n",
    "       'num_words_title', 'num_chars', 'chars_per_word', 'num_sentence']]\n",
    "\n",
    "y_train = train[['toxic', 'severe_toxic', 'obscene','threat', 'insult', 'identity_hate']]\n",
    "\n",
    "X_test = test[['num_words', 'num_stopwords', 'percentage_non_stop', 'num_caps','num_toxic_symbol', \n",
    "    'num_happy', 'num_sad',  'num_unique_words', 'punctuation', \n",
    "       'exclamation_marks', 'exclamation_marks_vs_length', 'num_newlines',\n",
    "       'num_words_title', 'num_chars', 'chars_per_word', 'num_sentence']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((3000, 16), (3000, 6), (3000, 16))"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape, y_train.shape, X_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.2 Testing base model\n",
    "Can change any other classifier with predict_proba funtion"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(3000, 16) (3000, 6) (3000, 16)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "{sklearn.multiclass.OneVsRestClassifier: (array([0.62819942, 0.63595899, 0.63848019, 0.71029088, 0.60796725,\n",
       "         0.60103605]),\n",
       "  0.6267779922022966,\n",
       "        0  1  2  3  4  5\n",
       "  0     0  0  0  0  0  0\n",
       "  1     1  0  1  1  1  0\n",
       "  2     1  0  1  1  1  0\n",
       "  3     0  0  0  0  0  0\n",
       "  4     1  0  1  1  1  0\n",
       "  ...  .. .. .. .. .. ..\n",
       "  2995  0  0  0  0  0  0\n",
       "  2996  1  0  1  1  1  1\n",
       "  2997  1  0  1  1  1  0\n",
       "  2998  0  0  0  1  0  0\n",
       "  2999  0  0  0  0  0  0\n",
       "  \n",
       "  [3000 rows x 6 columns]),\n",
       " sklearn.multioutput.ClassifierChain: (array([0.62819942, 0.64303991, 0.65810772, 0.65108602, 0.62441948,\n",
       "         0.62584232]),\n",
       "  0.635379060425371,\n",
       "        0  1  2  3  4  5\n",
       "  0     0  0  0  0  0  0\n",
       "  1     1  0  1  1  1  1\n",
       "  2     1  0  1  1  1  1\n",
       "  3     0  0  0  0  0  0\n",
       "  4     1  0  1  1  1  1\n",
       "  ...  .. .. .. .. .. ..\n",
       "  2995  0  0  0  0  0  0\n",
       "  2996  1  1  1  1  1  1\n",
       "  2997  1  0  1  1  1  1\n",
       "  2998  0  0  0  1  0  0\n",
       "  2999  0  0  0  0  0  0\n",
       "  \n",
       "  [3000 rows x 6 columns])}"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NB_models(X_train,y_train, X_test).model_base(ComplementNB(), proba_threshold = 0.2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.2 TESTING MLKNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>num_words</th>\n",
       "      <th>num_stopwords</th>\n",
       "      <th>percentage_non_stop</th>\n",
       "      <th>num_caps</th>\n",
       "      <th>num_toxic_symbol</th>\n",
       "      <th>num_happy</th>\n",
       "      <th>num_sad</th>\n",
       "      <th>num_unique_words</th>\n",
       "      <th>punctuation</th>\n",
       "      <th>exclamation_marks</th>\n",
       "      <th>exclamation_marks_vs_length</th>\n",
       "      <th>num_newlines</th>\n",
       "      <th>num_words_title</th>\n",
       "      <th>num_chars</th>\n",
       "      <th>chars_per_word</th>\n",
       "      <th>num_sentence</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>43</td>\n",
       "      <td>14</td>\n",
       "      <td>0.674419</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>41</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1</td>\n",
       "      <td>11</td>\n",
       "      <td>264</td>\n",
       "      <td>6.139535</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>17</td>\n",
       "      <td>1</td>\n",
       "      <td>0.941176</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>0.058824</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>112</td>\n",
       "      <td>6.588235</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>42</td>\n",
       "      <td>18</td>\n",
       "      <td>0.571429</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>39</td>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>233</td>\n",
       "      <td>5.547619</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>113</td>\n",
       "      <td>49</td>\n",
       "      <td>0.566372</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>82</td>\n",
       "      <td>21</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>4</td>\n",
       "      <td>7</td>\n",
       "      <td>622</td>\n",
       "      <td>5.504425</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>13</td>\n",
       "      <td>4</td>\n",
       "      <td>0.692308</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>67</td>\n",
       "      <td>5.153846</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2995</th>\n",
       "      <td>55</td>\n",
       "      <td>26</td>\n",
       "      <td>0.527273</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>44</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2</td>\n",
       "      <td>10</td>\n",
       "      <td>300</td>\n",
       "      <td>5.454545</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2996</th>\n",
       "      <td>6</td>\n",
       "      <td>2</td>\n",
       "      <td>0.666667</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>32</td>\n",
       "      <td>5.333333</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2997</th>\n",
       "      <td>82</td>\n",
       "      <td>39</td>\n",
       "      <td>0.524390</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>60</td>\n",
       "      <td>13</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1</td>\n",
       "      <td>14</td>\n",
       "      <td>402</td>\n",
       "      <td>4.902439</td>\n",
       "      <td>10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2998</th>\n",
       "      <td>78</td>\n",
       "      <td>33</td>\n",
       "      <td>0.576923</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>68</td>\n",
       "      <td>15</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>427</td>\n",
       "      <td>5.474359</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2999</th>\n",
       "      <td>16</td>\n",
       "      <td>3</td>\n",
       "      <td>0.812500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>15</td>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>107</td>\n",
       "      <td>6.687500</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3000 rows × 16 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      num_words  num_stopwords  percentage_non_stop  num_caps  \\\n",
       "0            43             14             0.674419         2   \n",
       "1            17              1             0.941176         1   \n",
       "2            42             18             0.571429         0   \n",
       "3           113             49             0.566372         5   \n",
       "4            13              4             0.692308         0   \n",
       "...         ...            ...                  ...       ...   \n",
       "2995         55             26             0.527273         3   \n",
       "2996          6              2             0.666667         0   \n",
       "2997         82             39             0.524390         9   \n",
       "2998         78             33             0.576923         2   \n",
       "2999         16              3             0.812500         0   \n",
       "\n",
       "      num_toxic_symbol  num_happy  num_sad  num_unique_words  punctuation  \\\n",
       "0                    0          0        0                41           10   \n",
       "1                    0          0        0                17           12   \n",
       "2                    0          0        0                39            6   \n",
       "3                    0          0        0                82           21   \n",
       "4                    0          0        0                13            5   \n",
       "...                ...        ...      ...               ...          ...   \n",
       "2995                 0          0        0                44            9   \n",
       "2996                 0          0        0                 6            0   \n",
       "2997                 0          0        0                60           13   \n",
       "2998                 0          0        0                68           15   \n",
       "2999                 0          0        0                15            6   \n",
       "\n",
       "      exclamation_marks  exclamation_marks_vs_length  num_newlines  \\\n",
       "0                     0                     0.000000             1   \n",
       "1                     1                     0.058824             0   \n",
       "2                     0                     0.000000             0   \n",
       "3                     0                     0.000000             4   \n",
       "4                     0                     0.000000             0   \n",
       "...                 ...                          ...           ...   \n",
       "2995                  0                     0.000000             2   \n",
       "2996                  0                     0.000000             0   \n",
       "2997                  0                     0.000000             1   \n",
       "2998                  0                     0.000000             1   \n",
       "2999                  0                     0.000000             1   \n",
       "\n",
       "      num_words_title  num_chars  chars_per_word  num_sentence  \n",
       "0                  11        264        6.139535             8  \n",
       "1                   3        112        6.588235             4  \n",
       "2                   2        233        5.547619             4  \n",
       "3                   7        622        5.504425             6  \n",
       "4                   2         67        5.153846             3  \n",
       "...               ...        ...             ...           ...  \n",
       "2995               10        300        5.454545             8  \n",
       "2996                0         32        5.333333             1  \n",
       "2997               14        402        4.902439            10  \n",
       "2998                8        427        5.474359             8  \n",
       "2999                8        107        6.687500             3  \n",
       "\n",
       "[3000 rows x 16 columns]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(3000, 16) (3000, 6) (3000, 16)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(array([0.53578828, 0.53913607, 0.5412102 , 0.5       , 0.51489129,\n",
       "        0.5       ]),\n",
       " 0.5301917590343149,\n",
       "       0  1  2  3  4  5\n",
       " 0     0  0  0  0  0  0\n",
       " 1     0  0  0  0  0  0\n",
       " 2     0  0  0  0  0  0\n",
       " 3     0  0  0  0  0  0\n",
       " 4     0  0  0  0  0  0\n",
       " ...  .. .. .. .. .. ..\n",
       " 2995  0  0  0  0  0  0\n",
       " 2996  0  0  0  0  0  0\n",
       " 2997  0  0  0  0  0  0\n",
       " 2998  0  0  0  0  0  0\n",
       " 2999  0  0  0  0  0  0\n",
       " \n",
       " [3000 rows x 6 columns])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NB_models(X_train,y_train, X_test).mlknn_model()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. TEST MODELING WITH COUNTVECTORIZER / TF-IDF\n",
    "This example will show you to use transformer such as countvectorizer or tf-idf to transform data and do modeling on this **transfromed data only**. Be aware that in the train data, I only have comment_text_clean column."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = train['comment_text_clean']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test = test['comment_text_clean']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3.1 Test base model with countervectorizer or tfidfvectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "This will only use transformed data, please check carefully\n",
      "(3000, 16959) (3000, 6) (3000, 16959)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "{sklearn.multiclass.OneVsRestClassifier: (array([0.87538449, 0.79671452, 0.88671897, 0.62996364, 0.87800578,\n",
       "         0.73652291]),\n",
       "  0.8635476867492047,\n",
       "        0  1  2  3  4  5\n",
       "  0     1  0  1  0  0  0\n",
       "  1     0  0  0  0  0  0\n",
       "  2     0  0  0  0  0  0\n",
       "  3     0  0  0  0  0  0\n",
       "  4     0  0  0  0  0  0\n",
       "  ...  .. .. .. .. .. ..\n",
       "  2995  0  0  0  0  0  0\n",
       "  2996  0  0  0  0  0  0\n",
       "  2997  1  1  1  1  1  1\n",
       "  2998  0  0  0  0  0  0\n",
       "  2999  0  0  0  0  0  0\n",
       "  \n",
       "  [3000 rows x 6 columns]),\n",
       " sklearn.multioutput.ClassifierChain: (array([0.87538449, 0.79671452, 0.89075291, 0.56272127, 0.87710276,\n",
       "         0.73298518]),\n",
       "  0.8628149594157882,\n",
       "        0  1  2  3  4  5\n",
       "  0     1  0  1  0  1  0\n",
       "  1     0  0  0  0  0  0\n",
       "  2     0  0  0  0  0  0\n",
       "  3     0  0  0  0  0  0\n",
       "  4     0  0  0  0  0  0\n",
       "  ...  .. .. .. .. .. ..\n",
       "  2995  0  0  0  0  0  0\n",
       "  2996  0  0  0  0  0  0\n",
       "  2997  1  1  1  0  1  1\n",
       "  2998  0  0  0  0  0  0\n",
       "  2999  0  0  0  0  0  0\n",
       "  \n",
       "  [3000 rows x 6 columns])}"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NB_models(X_train,y_train, X_test).model_base(ComplementNB(), proba_threshold = 0.2, use_transformer = CountVectorizer() )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3.2 Test base model with TfidfTransformer\n",
    "Tfidtransformer requires an extra step for countvectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "This will only use transformed data, please check carefully\n",
      "(3000, 16959) (3000, 6) (3000, 16959)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "{sklearn.multiclass.OneVsRestClassifier: (array([0.72704236, 0.64034792, 0.76610774, 0.53186298, 0.71780075,\n",
       "         0.63675034]),\n",
       "  0.721629687565422,\n",
       "        0  1  2  3  4  5\n",
       "  0     1  0  0  0  0  0\n",
       "  1     0  0  0  0  0  0\n",
       "  2     0  0  0  0  0  0\n",
       "  3     0  0  0  0  0  0\n",
       "  4     0  0  0  0  0  0\n",
       "  ...  .. .. .. .. .. ..\n",
       "  2995  0  0  0  0  0  0\n",
       "  2996  1  1  1  1  1  1\n",
       "  2997  1  1  1  1  1  1\n",
       "  2998  1  0  1  0  1  0\n",
       "  2999  0  0  0  0  0  0\n",
       "  \n",
       "  [3000 rows x 6 columns]),\n",
       " sklearn.multioutput.ClassifierChain: (array([0.72704236, 0.52412133, 0.80273551, 0.4996651 , 0.77812274,\n",
       "         0.49983154]),\n",
       "  0.730993762036645,\n",
       "        0  1  2  3  4  5\n",
       "  0     1  0  0  0  0  0\n",
       "  1     0  0  0  0  0  0\n",
       "  2     0  0  0  0  0  0\n",
       "  3     0  0  0  0  0  0\n",
       "  4     0  0  0  0  0  0\n",
       "  ...  .. .. .. .. .. ..\n",
       "  2995  0  0  0  0  0  0\n",
       "  2996  1  1  1  1  1  1\n",
       "  2997  1  1  1  0  1  0\n",
       "  2998  1  0  1  0  1  0\n",
       "  2999  0  0  0  0  0  0\n",
       "  \n",
       "  [3000 rows x 6 columns])}"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NB_models(X_train,y_train, X_test).model_base(ComplementNB(), proba_threshold = 0.2, use_transformer = TfidfTransformer())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3.3 Testing mlknn model with countervectorizer/ tfidvectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "This will only use transformed data, please check carefully\n",
      "(3000, 16959) (3000, 6) (3000, 16959)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(array([0.68609775, 0.55246277, 0.77191927, 0.5       , 0.7122321 ,\n",
       "        0.515625  ]),\n",
       " 0.6936872272741572,\n",
       "       0  1  2  3  4  5\n",
       " 0     0  0  0  0  0  0\n",
       " 1     0  0  0  0  0  0\n",
       " 2     0  0  0  0  0  0\n",
       " 3     0  0  0  0  0  0\n",
       " 4     0  0  0  0  0  0\n",
       " ...  .. .. .. .. .. ..\n",
       " 2995  0  0  0  0  0  0\n",
       " 2996  0  0  0  0  0  0\n",
       " 2997  0  0  0  0  0  0\n",
       " 2998  0  0  0  0  0  0\n",
       " 2999  0  0  0  0  0  0\n",
       " \n",
       " [3000 rows x 6 columns])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NB_models(X_train,y_train, X_test).mlknn_model(proba_threshold =0.5, k=10, s=1, use_transformer = CountVectorizer(), tag_transformer = False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3.4 Testing mlknn model with td-idftransformer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "This will only use transformed data, please check carefully\n",
      "(3000, 16959) (3000, 6) (3000, 16959)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(array([0.82277856, 0.63039731, 0.84571063, 0.60714286, 0.80533386,\n",
       "        0.515625  ]),\n",
       " 0.7962002002330165,\n",
       "       0  1  2  3  4  5\n",
       " 0     0  0  0  0  0  0\n",
       " 1     0  0  0  0  0  0\n",
       " 2     0  0  0  0  0  0\n",
       " 3     0  0  0  0  0  0\n",
       " 4     0  0  0  0  0  0\n",
       " ...  .. .. .. .. .. ..\n",
       " 2995  0  0  0  0  0  0\n",
       " 2996  0  0  0  0  0  0\n",
       " 2997  0  0  0  0  0  0\n",
       " 2998  0  0  0  0  0  0\n",
       " 2999  0  0  0  0  0  0\n",
       " \n",
       " [3000 rows x 6 columns])"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NB_models(X_train,y_train, X_test).mlknn_model(proba_threshold =0.5, k=10, s=1, use_transformer = TfidfTransformer())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Test LSA and tag data\n",
    "This example will use transformer such as countervectorizator, the transformed data will be used for LSA decomposition and then tagged along with the train data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = train[['num_words', 'num_stopwords', 'percentage_non_stop', 'num_caps','num_toxic_symbol', \n",
    "    'num_happy', 'num_sad',  'num_unique_words', 'punctuation', \n",
    "       'exclamation_marks', 'exclamation_marks_vs_length', 'num_newlines',\n",
    "       'num_words_title', 'num_chars', 'chars_per_word', 'num_sentence', 'comment_text_clean']]\n",
    "\n",
    "y_train = train[['toxic', 'severe_toxic', 'obscene','threat', 'insult', 'identity_hate']]\n",
    "\n",
    "X_test = test[['num_words', 'num_stopwords', 'percentage_non_stop', 'num_caps','num_toxic_symbol', \n",
    "    'num_happy', 'num_sad',  'num_unique_words', 'punctuation', \n",
    "       'exclamation_marks', 'exclamation_marks_vs_length', 'num_newlines',\n",
    "       'num_words_title', 'num_chars', 'chars_per_word', 'num_sentence', 'comment_text_clean']]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4.1 Test with countvectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "This will only use LSA reduced matrix with created features, please check carefully\n",
      "(3000, 21) (3000, 6) (3000, 21)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "{sklearn.multiclass.OneVsRestClassifier: (array([1., 1., 1., 1., 1., 1.]),\n",
       "  1.0,\n",
       "        0  1  2  3  4  5\n",
       "  0     0  0  0  0  0  0\n",
       "  1     0  0  0  0  0  0\n",
       "  2     0  0  0  0  0  0\n",
       "  3     0  0  0  0  0  0\n",
       "  4     0  0  0  0  0  0\n",
       "  ...  .. .. .. .. .. ..\n",
       "  2995  0  0  0  0  0  0\n",
       "  2996  0  0  0  0  0  0\n",
       "  2997  1  0  1  0  1  0\n",
       "  2998  0  0  0  0  0  0\n",
       "  2999  0  0  0  0  0  0\n",
       "  \n",
       "  [3000 rows x 6 columns]),\n",
       " sklearn.multioutput.ClassifierChain: (array([1., 1., 1., 1., 1., 1.]),\n",
       "  1.0,\n",
       "        0  1  2  3  4  5\n",
       "  0     0  0  0  0  0  0\n",
       "  1     0  0  0  0  0  0\n",
       "  2     0  0  0  0  0  0\n",
       "  3     0  0  0  0  0  0\n",
       "  4     0  0  0  0  0  0\n",
       "  ...  .. .. .. .. .. ..\n",
       "  2995  0  0  0  0  0  0\n",
       "  2996  0  0  0  0  0  0\n",
       "  2997  1  0  1  0  1  0\n",
       "  2998  0  0  0  0  0  0\n",
       "  2999  0  0  0  0  0  0\n",
       "  \n",
       "  [3000 rows x 6 columns])}"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NB_models(X_train,y_train, X_test).model_base(RandomForestClassifier(), proba_threshold = 0.5, use_transformer = TfidfVectorizer(),  tag_transformer = True )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4.2 Test with tf-idf transformer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "This will only use LSA reduced matrix with created features, please check carefully\n",
      "(3000, 21) (3000, 6) (3000, 21)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "{sklearn.multiclass.OneVsRestClassifier: (array([1., 1., 1., 1., 1., 1.]),\n",
       "  1.0,\n",
       "        0  1  2  3  4  5\n",
       "  0     0  0  0  0  0  0\n",
       "  1     0  0  0  0  0  0\n",
       "  2     0  0  0  0  0  0\n",
       "  3     0  0  0  0  0  0\n",
       "  4     0  0  0  0  0  0\n",
       "  ...  .. .. .. .. .. ..\n",
       "  2995  0  0  0  0  0  0\n",
       "  2996  0  0  0  0  0  0\n",
       "  2997  1  0  1  0  1  0\n",
       "  2998  0  0  0  0  0  0\n",
       "  2999  0  0  0  0  0  0\n",
       "  \n",
       "  [3000 rows x 6 columns]),\n",
       " sklearn.multioutput.ClassifierChain: (array([1., 1., 1., 1., 1., 1.]),\n",
       "  1.0,\n",
       "        0  1  2  3  4  5\n",
       "  0     0  0  0  0  0  0\n",
       "  1     0  0  0  0  0  0\n",
       "  2     0  0  0  0  0  0\n",
       "  3     0  0  0  0  0  0\n",
       "  4     0  0  0  0  0  0\n",
       "  ...  .. .. .. .. .. ..\n",
       "  2995  0  0  0  0  0  0\n",
       "  2996  0  0  0  0  0  0\n",
       "  2997  1  0  1  0  1  0\n",
       "  2998  0  0  0  0  0  0\n",
       "  2999  0  0  0  0  0  0\n",
       "  \n",
       "  [3000 rows x 6 columns])}"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NB_models(X_train,y_train, X_test).model_base(RandomForestClassifier(), proba_threshold = 0.5, use_transformer = TfidfTransformer(),  tag_transformer = True )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4.3 Testing mlknn model tagged with countvectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "This will only use LSA reduced matrix with created features, please check carefully\n",
      "(3000, 21) (3000, 6) (3000, 21)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(array([0.53578828, 0.53913607, 0.5412102 , 0.5       , 0.51489129,\n",
       "        0.5       ]),\n",
       " 0.5301917590343149,\n",
       "       0  1  2  3  4  5\n",
       " 0     0  0  0  0  0  0\n",
       " 1     0  0  0  0  0  0\n",
       " 2     0  0  0  0  0  0\n",
       " 3     0  0  0  0  0  0\n",
       " 4     0  0  0  0  0  0\n",
       " ...  .. .. .. .. .. ..\n",
       " 2995  0  0  0  0  0  0\n",
       " 2996  0  0  0  0  0  0\n",
       " 2997  0  0  0  0  0  0\n",
       " 2998  0  0  0  0  0  0\n",
       " 2999  0  0  0  0  0  0\n",
       " \n",
       " [3000 rows x 6 columns])"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NB_models(X_train,y_train, X_test).mlknn_model(proba_threshold = 0.5, use_transformer = TfidfVectorizer(),  tag_transformer = True )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4.4 Testing mlknn model tagged with td-idftransformer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "This will only use LSA reduced matrix with created features, please check carefully\n",
      "(3000, 21) (3000, 6) (3000, 21)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(array([0.53578828, 0.53913607, 0.5412102 , 0.5       , 0.51489129,\n",
       "        0.5       ]),\n",
       " 0.5301917590343149,\n",
       "       0  1  2  3  4  5\n",
       " 0     0  0  0  0  0  0\n",
       " 1     0  0  0  0  0  0\n",
       " 2     0  0  0  0  0  0\n",
       " 3     0  0  0  0  0  0\n",
       " 4     0  0  0  0  0  0\n",
       " ...  .. .. .. .. .. ..\n",
       " 2995  0  0  0  0  0  0\n",
       " 2996  0  0  0  0  0  0\n",
       " 2997  0  0  0  0  0  0\n",
       " 2998  0  0  0  0  0  0\n",
       " 2999  0  0  0  0  0  0\n",
       " \n",
       " [3000 rows x 6 columns])"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NB_models(X_train,y_train, X_test).mlknn_model(proba_threshold = 0.5, use_transformer = TfidfTransformer(),  tag_transformer = True )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
