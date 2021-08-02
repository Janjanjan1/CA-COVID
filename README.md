
# CA-COVID - Janeet Bajracharya
While ODE-IVP Models have dominated Epidemiological Modeling for their superior performance and flexibility, they fail to capture the microscopic (relatively speaking) interactions between populations that account for the spread of disease. Acknowledging the nature of infectious disease and to be specific COVID, using a Cellular Automaton can be extremely useful to model the spread given certain initial conditions. Loosely using the model described in [A Data-driven Understanding of COVID-19 Dynamics Using Sequential Genetic Algorithm Based Probabilistic Cellular Automata – Sayantari Ghosha, Saumik Bhattacharya] this project contains two parts:  
1) Model with Genetic Algorithm 
2) Model with Visualizations and Data Production

The model runs on Daily Case Numbers in the form of a CSV. To train the model first to get the most optimal parameters run CA-GA.py with a file named data.csv in the same folder. Have a column in the CSV named Confirmed with Daily Active Cases. If need be normalize the daily Cases if the Number of Cases at any point exceed 3100.

To calibrate the number of Susceptible People in the population please refer to the Logarithmic Scale mentioned in the research paper mentioned above. 

Here are some of the examples produced by the model, calibrated to data for Nepal:

![nice](https://user-images.githubusercontent.com/73707418/127877375-c2a80a61-440f-4ff0-b17e-5204e788aebb.gif)
![modeld vs real](https://user-images.githubusercontent.com/73707418/127877482-1a12e188-d7cc-4acf-8db4-726c0beeede8.png)

