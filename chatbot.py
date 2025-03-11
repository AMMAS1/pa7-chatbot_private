# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
# Update: 2025-01 for Winter 2025 (Xuheng Cai)
######################################################################
import util
from pydantic import BaseModel, Field
import re
import numpy as np
import json

##### todo
# - edit the system prompt for llm prompting mode
# - make sure system prompt works for llm_programming mode

##### questions
# - what exactly is the persona? can it just be movie bot?
# - repetition in llm prompting mode?
# - what's the diff btn llm prompting and llm programming mode?
# - in programming mode? don't we just use the same prompt from part 2 for the bot or what's the diff?
# - what happens when the bot llm use to temperature at some point during grading?
# - how many movies do you use for the weighed score in the item-item collaborative filtering?
# - what to do when the user says no to the recommendations? (faq says up to u but do we reset the whole thing?)
# - what to do when the user says they alr watched a recommended movie do i take it as a yes and rec more?
# - what to do if the user rated a recommended movie? do i reset the reccommendations list?
# - can the user still rate movies after the first 5 or while the bot is reccommending?
# - what to do if the user rates a movie they already rated?
# - what if the user rates two movies at the same time
# - what if the user says a movie w/o quotation marks or wrong?

##### assumptions
# - if there are multiple movies with the same name, we will ask the user to specify which one they are talking about (as faq)


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'moviebot'

        self.llm_enabled = llm_enabled
        # this is for llm programming to swtich from json to normal mode after failing n times
        self.trials = 5
        self.messages = []

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        self.user_ratings = np.zeros(self.ratings.shape[0])
        self.recs = []
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "How can I help you?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if self.llm_enabled:
            client = util.load_together_client()
            chat_completion = client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": self.llm_system_prompt(),
                }] + self.messages + [{
                    "role": "user",
                    "content": line,
                }],
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                max_tokens=256,
            )
            self.messages.append({
                "role": "user",
                "content": line,
            })
            self.messages.append({
                "role": "assistant",
                "content": chat_completion.choices[0].message.content,
            })
            return chat_completion.choices[0].message.content
        else:
            response = ""
            if len(self.recs) == 0: #  we still below 5 ratings
                # extract the title
                potential_titles = self.extract_titles(self.preprocess(line))
                if len(potential_titles) == 0:
                    return "I'm sorry, I couldn't find any movie titles in your input"
                elif len(potential_titles) > 1:
                    return "I'm sorry, I found multiple movie titles in your input"
                title = potential_titles[0]

                # find the movie by title
                movie_indices = self.find_movies_by_title(title)
                if len(movie_indices) == 0:
                    return "I'm sorry, I couldn't find any movies with the title {}".format(title)
                elif len(movie_indices) > 1:
                    return "I found multiple movies with the title {}, here are the first {}: {}\n Which one are you talking about?".format(title, min(5, len(movie_indices)), ", ".join([self.titles[i][0] for i in movie_indices[:5]]))
                movie_index = movie_indices[0]

                # extract sentiment
                sentiment = self.extract_sentiment(self.preprocess(line))
                if sentiment == 1:
                    self.user_ratings[movie_index] = 1
                    positive_responses = [
                        "You liked {}!".format(self.titles[movie_index][0]),
                        "So you liked {}!".format(self.titles[movie_index][0]),
                        "You enjoyed {}!".format(self.titles[movie_index][0]),
                        "Ah, I am glad you found {} good!".format(self.titles[movie_index][0]),
                        "You loved {}!".format(self.titles[movie_index][0])
                    ]
                    response = np.random.choice(positive_responses)
                elif sentiment == 0:
                    neutral_responses = [
                        "You felt neutral about {}.".format(self.titles[movie_index][0]),
                        "{} was just okay for you.".format(self.titles[movie_index][0]),
                        "It seems {} didn’t leave a strong impression.".format(self.titles[movie_index][0]),
                        "You're indifferent about {}.".format(self.titles[movie_index][0]),
                        "{} was neither good nor bad for you.".format(self.titles[movie_index][0])
                    ]
                    response = np.random.choice(neutral_responses)

                else:
                    negative_responses = [
                        "You didn’t like {}!".format(self.titles[movie_index][0]),
                        "So {} wasn’t your favorite.".format(self.titles[movie_index][0]),
                        "Oh, you disliked {}!".format(self.titles[movie_index][0]),
                        "Not a fan of {}?".format(self.titles[movie_index][0]),
                        "You didn’t enjoy {}.".format(self.titles[movie_index][0])
                    ]
                    self.user_ratings[movie_index] = -1
                    response = np.random.choice(negative_responses)

            if np.count_nonzero(self.user_ratings) < 5: # we still need more ratings ask user for more
                questions = [
                    "Tell me more movies you liked!",
                    "What other movies can you tell me?",
                    "What other movies have you watched?",
                    "Any other movies you've seen recently?",
                    "What other movies do you have thoughts on?",
                    "What else have you watched?",
                    "Do you have any other favorites or least favorites?",
                    "Let’s talk about another movie!"
                ]
                response += " " + np.random.choice(questions)
            else:
                if len(self.recs) > 0: # if recs is already filled then last msg was asking if the user wants more recs
                    yesses = ["yes", "yeah", "sure", "ok", "okay", "yep", "y", "yea", "yup", "of course", "please", "more", "another"]
                    nos = ["no", "nope", "nah", "n", "not really", "not", "not now", "not yet", "not today", "not this time", "no thanks", "no thank you", "no more","im good", "i'm good"]
                    # remove punctuation and lowercase the line
                    line_clean = re.sub(r'[^\w\s]', '', line).lower()
                    if any(yes in line_clean for yes in yesses):
                        # recommend more
                        pass  
                    elif any(no in line_clean for no in nos):
                        return self.goodbye()
                        # finish
                    else: # keep asking till the user responds
                        return "I'm sorry, I didn't understand your response. Do you want more recommendations?"
                else: # if recs is empty and we have 5 ratings that means we just got the fifth movie.
                    self.recs = self.recommend(self.user_ratings, self.ratings)[::-1] # we reverse the list to get the highest rated movies first when popping
                
                # respond with another recommendation if user responded with yes or it's first time recommending
                responses = [
                    "Given what you said, I have some recommendations for you: ",
                    "Based on your input, I have something for you to watch: ",
                    "Let me reccomend this to you then: ",
                    "Based on out conversation, I think you would like: ",
                    ]

                response += " " + np.random.choice(responses) + self.titles[self.recs.pop()][0]

                follow_ups = [
                    "Do you want more recommendations?",
                    "Would you like more recommendations?",
                    "Do you want another recommendation?",
                    "Want another recommendation?",
                ]

                response += " " + np.random.choice(follow_ups)

                response = response.strip()



        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        # all the substrings between the quotation marks
        return [title for title in re.findall(r'"([^"]*)"', preprocessed_input)]

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        
        if self.llm_enabled:
            class movietitle(BaseModel):
                English: str = Field(description="The movie title in English")
            for i in range(self.trials):
                try:
                    response = self.json_llm_call("Translate movie title from German, Spanish, and French, Danish, and Italian to ENGLISH. Only Respond with the name in english in json nothing else",
                            title, movietitle.model_json_schema())
                    # check if the response is a valid emotion json object
                    if not response \
                    or not (response := json.loads(response)) \
                    or not isinstance(response, dict) \
                    or "English" not in response \
                    or not isinstance(response["English"], str):
                        raise ValueError("Invalid emotion response")
                    title = response["English"]
                    break
                except:
                    pass
            
        results = []
        for i in range(len(self.titles)):    
            # remove The or an or a from the beginning of the title but only if it is the first word
            if re.match(r"^.*?, (The|A|An) (\(\d+\))", self.titles[i][0]):
                movietitle, article, year = re.match(r'^(.+?), (The |A |An )(\(\d+\))$', self.titles[i][0]).groups()
            else:
                article, movietitle, year = re.match(r'^(The |A |An )?(.+?) ?(\(\d+\))?$', self.titles[i][0]).groups()
            pattern = rf"^({article if article else ''})?({re.escape(movietitle)})( {re.escape(year) if year else ''})?$"
            if re.match(pattern, title):
                results.append(i)
        return results

    def stem(self, word):
        stems = [word]
        if word.endswith("ing"):
            stems.append(word[:-3])
        if word.endswith("ed"):
            stems.append(word[:-2]) # enjoyed -> enjoy
            stems.append(word[:-1]) # liked -> like
        if word.endswith("ly"):
            stems.append(word[:-2]) # happily -> happy
        if word.endswith("s"): # plural form
            stems.append(word[:-1]) # likes -> like
        return stems

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        negation_words = ["not", "no", "never", "neither", "nor", "none", "nowhere", "nothing"]
        score = 0
        negation_flag = 1
        # remove between quotation marks and replace them with a space
        preprocessed_input = re.sub(r'"([^"]*)"', " ", preprocessed_input)
        for word in preprocessed_input.split():
            # stemming the word
            word = word.lower()
            # check if a word is negation word
            if word in negation_words or word.endswith("n't"):
                negation_flag = -negation_flag
            else:
                for stem in self.stem(word):
                    if stem in self.sentiment:
                        score += negation_flag * {"neg": -1, "pos": 1}[self.sentiment[stem]]
                        break
        return np.clip(score, -1, 1)

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.zeros_like(ratings)
        binarized_ratings[ratings > threshold] = 1
        binarized_ratings[(ratings > 0) & (ratings <= threshold)] = -1

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9) # to avoid division by zero
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        recommendations = []
        unrated_indices = np.where(user_ratings == 0)[0]
        # user_rated_matrix is a smaller matrix with only the movies that the user has rated
        user_rated_matrix = ratings_matrix[user_ratings != 0, :] # rated movies x users
        user_unrated_matrix = ratings_matrix[user_ratings == 0, :] # unrated movies x users

        # note that this normalization is only for cosine similarity we are not normalizing the scores
        norms1 = np.linalg.norm(user_rated_matrix, axis=1, keepdims=True)
        norms2 = np.linalg.norm(user_unrated_matrix, axis=1, keepdims=True)
        similarity_matrix = np.dot(user_rated_matrix, user_unrated_matrix.T) / (np.dot(norms1, norms2.T)+1e-9) # (rated movies x users) x (users x unrated movies) = (rated movies x unrated movies)
        
        # now we have the similarity matrix, we can use it to fill the missing movies ratings with weighted average
        interpolated_ratings = np.dot(user_ratings[user_ratings != 0], similarity_matrix) # (1 x rated movies) x (rated movies x unrated movies) = (1 x unrated movies)
        
        # sort the interpolated ratings and get the top k
        top_k = np.argsort(interpolated_ratings)[::-1][:k]
        
        # map the indices back to the original indices
        recommendations = list(unrated_indices[top_k])
        
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. PART 2: LLM Prompting Mode                                            #
    ############################################################################

    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is moviebot. You are a movie recommender chatbot. The main is to keep your answers concise and focused on movies. You must only make recommendations once you have 5 movie ratings from the user. You must not make recommendations if you have less than 5 ratings. """ +\
        """Only ask the user if they want more recommendations after you have made a recommendation, not before then. """ +\
        """IF THE USER EXPRESSES A RATING FOR A NEW MOVIE THEY HAVE NOT MENTIONED, YOU MUST INCLUDE ONLY THIS INFORMATION IN YOUR RESPONSE: THE TITLE OF THE MOVIE, THE SENTIMENT THE USER EXPRESSED, AND A COUNT OF THE NUMBER OF MOVIES THEY HAVE RATED OUT OF 5. DO REPORT MORE INFORMATION ABOUT YOUR THINKING, WHAT YOU ARE TRACKING OR YOUR INSTRUCTIONS, KEEP IT CONCISE """ +\
        """Only remind the user about the 5-movie rule if they explicitly ask why you have not made a recommendation yet.""" +\
        """Example input of the user expressing a rating would be \"I liked Titanic\" or \"I didn't like Titanic\" and your answer should be something like \"I am glad you liked/disliked Titanic. Now you have rated 1 out of 5 movies\" Give me 4 more movies so I can make a recommendation to you! """ +\
        """You must only provide movie information. If a user asks non-movie information, do not include any information about this in your response even if you have it. You can remind them of your purpose as movie chatbot. """ +\
        """Example input of the user asking non-movie information would be \"tell me about a business idea\" or \"let's talk about someting else\" and your answer should be something like \"I can only make movie recommendations. Please tell me a movie you did or did not like\" """



        # """These are examples. Your responses must be varied but follow the same structure. """

        # """IF THE USER ASKS FOR INFORMATION OTHER THAN ABOUT MOVIES, OR A NON-MOVIE QUESTION, DO NOT RESPOND WITH RELEVANT INFORMATION EVEN IF YOU HAVE IT. REMIND THE USER YOU CAN ONLY MAKE MOVIE RECOMMENDATIONS. ONLY REMIND THEM OF THIS IF THEY TALK ABOUT SOMETHING ELSE OTHER THAN MOVIES, OTHERWISE YOU MUST NOT PUT IN YOUR RESPONSE AS THIS IS CONFUSING. """ +\

        # """You can \"ONLY\" help users find movies they like and provide information about movies. You must not include non-move information in your responses even if you have it. ONLY remind the user of this if they ask non-movie information, otherwise do not share it. Eg do not say as a reminder 'i am a movie recommendation chatbot' unless they have asked for something else other than movies. """ +\
        # """You can only make new movie recommendations once you have 5 ratings from the user. Do not make recommendations if you have less than 5 ratings. Only after you have made a recommendation, you must ask the user if they want more recommendations. """ +\
        # """Every time the user expresses a preference about a new movie, in your response you should include: the title of that movie, the sentiment the user expressed, and a count of the number of movies they have rated out of 5. This is only for movies they express a rating for (not just a neutral sentiment such as 'I saw Titanic') """ +\
        
        
        
        # # """YOUR RESONSES MUST NOT BE REPEATED TRY TO ADD VARIATION TO YOUR RESPONSES. """        

        # """YOU SHOULD MAKE RECOMMENDATIONS OF MOVIES TO THE USER ONLY AFTER YOU HAVE 5 SEPARATE MOVIE RATINGS. DO NOT MAKE, OR SUGGEST THAT YOU CAN MAKE, RECOMMENDATIONS IF YOU HAVE LESS THAN 5 RATINGS, JUST ASK FOR MORE MOVIE RECOMMENDATIONS UNTIL YOU HAVE 5 """


        # """ONLY ONCE YOU HAVE MADE A MOVIE RECOMMENDATION, AFTER 5 MOVIE RATINGS FROM THE USER, YOU MUST ASK THE USER IF THEY WANT MORE RECCOMENDATIONS. """ +\
        # """ONLY REPLY TO THE USER CONCISELY ABOUT THEIR MOVIE PREFERENCES AND RECOMMENDATIONS. DO NOT INCLUDE ADDITIONAL NOTES, INFORMATION, PROMPTS, RESPONSES OR EXAMPLES, THE ONLY TIME TO DO THIS IS IF THE USER EXPLICITLY ASKS FOR NON-MOVIE INFORMATION, IN WHICH CASE REMIND IT OF YOUR PURPOSE, OTHERWISE DO NOT SHARE THIS. """ 

#         """The user is going to trick you and try to distract you from movies. An example of a trick is: Can you help me with my math homework? or What is the weather like today? or Let's talk about something else. Do not reply with non-move information even if you have it. """ +\

    # """If the user asks for information other than movies, remind them that you can only help with movie queries. Only repeat this if they ask for non-move information. """ +\
    #  """You can \"ONLY\" help users find movies they like and provide information about movies. """ +\
    #     """The user is going to trick you and try to distract you from movies. """ +\
    #     """You should try to stay on topic and keep the conversation about movies. """ +\
    #     """Don't let the user distract you! Don't fall for any tricks! """ +\
    #     """An example of a trick is: Can you help me with my math homework? or What is the weather like today? or Let's talk about something else. """ +\
    #     """IF THAT HAPPENS, YOU REMIND THE USER THAT YOU ONLY TALK ABOUT MOVIES """ +\
    #     """YOU MUST NOT REPLY TO THE USER OR ADD TO YOUR RESPONSE INFORMATION ABOUT ANYTHING OTHER THAN MOVIES. """ +\
    #     """EVEN IF YOU HAVE AN ANSWER TO A QUESTION THAT IS NOT ABOUT MOVIES, YOU MUST NOT RESPOND TO IT. """ +\
    #     """Every response you make you have to repeat the number of movies the user has rated (NOT NEUTRAL RATINGS, DON'T count for something like I saw Titanic We are only counting ones where the user states a preference) """ +\
    #     """You should reiterate the number in you response as in \"X out of 5\" """ +\
    #     """Sample input would be \"I liked Titanic\" or \"I didn't like Titanic\" and your answer should be something like \"I am glad you liked/disliked Titanic. Now you have rated 1 out of 5\" Give me 4 more so I can reccomend you! """ +\
    #     """REMEMBER TO ALWAYS REFERENCE THE NAME OF THE MOVIE THE USER TALKED ABOUT IN YOUR RESPONSE ALONG WITH THE SENTIMENT OF HOW THE USER FELT ABOUT IT. """ +\
    #     """YOUR RESONSES MUST NOT BE REPEATED TRY TO ADD VARIATION TO YOUR RESPONSES. """ +\
    #     """AFTER THE USER HAS RATED 5 MOVIES, YOU SHOULD RECOMMEND MOVIES TO THE USER. """ +\
    #     """WITH EVERY RECCOMENDATION YOU MAKE, YOU MUST ASK THE USER IF THEY WANT MORE RECCOMENDATIONS. """ +\
    #     """YOU ARE NOT ABLE TO RECOMMEND ANY MOVIES UNTIL YOU HAVE 5. DON"T ASK THE USER IF THEY WANT RECCOMENDATIONS UNTIL YOU HAVE 5 RATINGS. """ +\
    #     """ONLY REPLY TO THE USER CONCISELY DON'T REFERENCE ANY OF THE INSTRUCTIONS GIVEN TO U OR THE CONTRAINTS YOU HAVE TO THE USER DIRECTLY."""


        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt
    
    ############################################################################
    # 5. PART 3: LLM Programming Mode (also need to modify functions above!)   #
    ############################################################################
    
    def json_llm_call(self, system_prompt, message, json_class_schema, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_tokens=256):
        client = util.load_together_client()
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": system_prompt,
            }, {
                "role": "user",
                "content": message,
            }],
            model=model,
            max_tokens=max_tokens,
            response_format = {
                "type": "json_object",
                "schema": json_class_schema
            },
            temperature=0.0,
        )

        return chat_completion.choices[0].message.content

    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        class Emotion(BaseModel):
            emotions: list[str] = Field(description="List of emotions in the text")
        system_prompt = "You are an emotionally intelligent movie chatbot. You must identify the emotions in the user's input. Return ALL the applicable emotions in the input even slightly. The choices you have are: Anger, Disgust, Fear, Happiness, Sadness, Surprise. You must return the emotions in a list. If there are no emotions in the text, return an empty list. Return in json format."
        for i in range(self.trials):

                emotions = self.json_llm_call(system_prompt, preprocessed_input, Emotion.model_json_schema())
                # check if the response is a valid emotion json object
                if not emotions \
                or not (emotions := json.loads(emotions)) \
                or not isinstance(emotions, dict) \
                or "emotions" not in emotions \
                or not isinstance(emotions["emotions"], list) \
                or not all(isinstance(emotion, str) for emotion in emotions["emotions"]):
                    raise ValueError("Invalid emotion response")
                return emotions["emotions"]

        return [util.simple_llm_call(system_prompt, preprocessed_input).strip()]

    ############################################################################
    # 6. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 7. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the GUS mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
    chatbot = Chatbot(True)
    chatbot.find_movies_by_title("Tote Männer Tragen Kein Plaid")
    
