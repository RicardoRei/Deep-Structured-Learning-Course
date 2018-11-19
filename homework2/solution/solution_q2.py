from sequence_decoder import SequenceDecoder
import numpy as np

weather = ["Sunny", "Windy", "Rainy"]
activities = ["Surf", "Beach", "Videogame", "Study"]

emission_probabilities = np.array([[0.4, 0.5, 0.1],
                                   [0.4, 0.1, 0.1],
                                   [0.1, 0.2, 0.3],
                                   [0.1, 0.2, 0.5]])

transition_probabilities = np.array([[0.7, 0.3, 0.2],
                                     [0.2, 0.5, 0.3],
                                     [0.1, 0.2, 0.5]])

observations = ["Videogame",
                "Study",
                "Study",
                "Surf",
                "Beach",
                "Videogame",
                "Beach"]

initial_weather = "Rainy"
final_weather = "Sunny"

x = [activities.index(observation) for observation in observations]
emission_scores = np.log(emission_probabilities[x, :])
transition_scores = np.array([np.log(transition_probabilities)
                              for _ in range(len(observations))])
initial_scores = np.log(
    transition_probabilities[:, weather.index(initial_weather)])
final_scores = np.log(
    transition_probabilities[weather.index(final_weather), :])

decoder = SequenceDecoder()
best_path, _ = decoder.run_viterbi(initial_scores,
                                   transition_scores,
                                   final_scores,
                                   emission_scores)

print("Best sequence: %s" % " -> ".join([weather[i] for i in best_path]))

posteriors = decoder.run_forward_backward(initial_scores,
                                          transition_scores,
                                          final_scores,
                                          emission_scores)
print(posteriors)
min_risk_path = posteriors.argmax(1)

print("Min risk sequence: %s" % " -> ".join(
    [weather[i] for i in min_risk_path]))

bet = 1. # In Euros.
print("Expected profit: %f Euros." % ((2*posteriors.max(1) - 1).sum() * bet))
#print("Expected profit of always-raining: %f Euros." % (
#    (2*posteriors[:, weather.index('Rainy')] - 1).sum() * bet))

min_risk_path_no_observations = []
current = transition_probabilities[:, weather.index(initial_weather)]
for _ in range(len(observations)):
    print(current)
    min_risk_path_no_observations.append(current.argmax())
    current = transition_probabilities.dot(current)
print("Min risk sequence without observations: %s" % " -> ".join(
    [weather[i] for i in min_risk_path_no_observations]))
print("Expected profit without observations: %f Euros." % (
    (2*posteriors[range(len(observations)),
                  min_risk_path_no_observations] - 1).sum() * bet))
