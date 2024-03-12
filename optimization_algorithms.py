import numpy as np
import csv

def evaluate_fitness(ANN, data, method = 'mse'):
    features, labels = data
    labels_pred = ANN.predict_rot(features.to_numpy(),\
                         rotational_invariance = False,\
                         output = 'invariant') # Run the trained model
    
    if method == 'mse':
        return np.square(labels.to_numpy() - labels_pred).mean()
    else:
        print("Select a valid fitness-calculation method")

def gridsearch(f, niter = 30, bounds = [], f_args = []):
    X = np.zeros((niter, np.shape(bounds)[0]))
    Fitness = []

    # Create range of parameters
    for i in range(len(bounds)):
        # X[:, i] = np.random.randint(bounds[i][0], bounds[i][1], size = niter)
        if isinstance(bounds[i][0], int) or bounds[i][0].is_integer():
            X[:, i] = np.random.randint(bounds[i][0], bounds[i][1], size = niter)
        else:
            X[:, i] = np.random.uniform(low = bounds[i][0], high = bounds[i][1], size = niter)

    # with open("./train/opt/gridsearch_X.csv", "w+") as output:
    #     # output.write(str(X))
    #     wr = csv.writer(output)
    #     wr.writerows(X)

    for i in range(niter):
        print("Iteration: %i/%i\n"%(i, niter))
        # Evaluate current x
        fitness = f(X[i,:], f_args)
        Fitness.append(fitness)

    return X, Fitness

    #     with open("./train/opt/gridsearch_training_loss.csv", "w+") as output:
    #         wr = csv.writer(output)
    #         wr.writerows(Loss_train)
    #     with open("./train/opt/gridsearch_validation_loss.csv", "w+") as output:
    #         wr = csv.writer(output)
    #         wr.writerows(Loss_val)


###############################################
# MONOTONIC BASIN HOPPING
###############################################
# Function to change step in basin hoping
class MyTakeStep(object):
    def __init__(self, Nimp, bnds):
        self.Nimp = Nimp
        self.bnds = bnds
    def __call__(self, x):
        self.call(x)
        return x

    def call(self, x, magnitude):
        x2 = np.zeros(len(x))
        for j in range(len(x)):
            x2[j] = x[j] + magnitude* np.random.uniform(self.bnds[j][0] - x[j], \
                                    self.bnds[j][1] - x[j], 1)[0]

        # x[0] += np.random.normal(-1e3, 1e3, 1)[0] # velocity magnitude
        # x[1:3] += np.random.normal(-0.2, 0.2, 1)[0] # angle
        # x[3] += np.random.normal(-1e3, 1e3, 1)[0] # velocity magnitude
        # x[4:6] += np.random.normal(-0.2, 0.2, 1)[0] # angle
        # x[6] += np.random.normal(-30, 30, 1)[0] # time in days
        # x[7] += np.random.normal(-30, 30, 1)[0] # initial mass
        # x[8] += np.random.normal(- AL_BF.days2sec(30), AL_BF.days2sec(30), 1)[0] # transfer time 
        # for i in range(self.Nimp):
        #     x[9+i*3] += np.random.normal(-0.1, 0.1, 1)[0]
        #     x[9+i*3+1 : 9+i*3+3] += np.random.normal(-0.5, 0.5, 2)
        
        return x2

def print_fun(f, x, accepted):
    """
    print_fun: choose what to print after each iteration
    INPUTS: 
        x: current value of the decision vector
        f: function value
        accepted: iteration improves the function = TRUE. Otherwise FALSE.
    OUTPUTS: none
    """
    print("Change iteration",accepted)
    print("#####################################################################")
    # print('t',time.time() - start_time)


def print_sol(Best, bestMin, n_itercounter, niter_success):
    """
    print_sol: 
    INPUTS: 
        
    OUTPUTS: none
    """
    print("Number of iterations", n_itercounter)
    print("Number of iterations with no success", niter_success)
    print("Minimum", bestMin)
    print("x", Best)
    print("#####################################################################")


def check_feasibility(x, bnds):
    feasible = True
    for j in range(len(x)): 
        if bnds[j][0] <0:
            tol = 1.05
        else:
            tol = 0.95
        if ( x[j] < tol*bnds[j][0] ) or ( x[j] > 1.05*bnds[j][1] ): # 1.05* to have tolerance
            feasible = False
            print(j, "Within bounds?", "min", bnds[j][0], "value",x[j], "max",bnds[j][1])
    # print(feasible)

    return feasible

def check_constraints(x, cons):
    value = cons['fun'](x)
    if value == 0:
        return True
    else:
        return False

def MonotonicBasinHopping(f, x, take_step, *args, **kwargs):
    """
    Step for jump is small, as minimum cluster together.
    Jump from current min until n_no improve reaches the lim, then the jump is random again.
    """

    niter = kwargs.get('niter', 100)
    niter_success = kwargs.get('niter_success', 50)
    niter_local = kwargs.get('niter_local', 50)
    bnds = kwargs.get('bnds', None)
    cons =  kwargs.get('cons', 0)
    jumpMagnitude_default =  kwargs.get('jumpMagnitude', 0.1) # Small jumps to search around the minimum
    tolLocal = kwargs.get('tolLocal', 1e2)
    tolGlobal = kwargs.get('tolGobal', 1e-5)
    
    n_itercounter = 1
    n_noimprove = 0

    Best = x
    bestMin =  f(x)
    previousMin = f(x)
    jumpMagnitude = jumpMagnitude_default

    while n_itercounter < niter:
        n_itercounter += 1
        
        # Change decision vector to find one within the bounds
        feasible = False
        while feasible == False and bnds != None:
            x_test = take_step.call(x, jumpMagnitude)
            feasible = check_feasibility(x_test, bnds)
            # if feasible == True:
            #     feasible = check_constraints(x, cons)

        
        # Local optimization 
        # solutionLocal = spy.minimize(f, x, method = 'COBYLA', constraints = cons, options = {'maxiter': niter_local} )
        if type(cons) == int:
            solutionLocal = spy.minimize(f, x_test, method = 'SLSQP', \
                tol = tolLocal, bounds = bnds, options = {'maxiter': niter_local} )
        else:
            solutionLocal = spy.minimize(f, x_test, method = 'SLSQP', \
                tol = tolLocal, bounds = bnds, options = {'maxiter': niter_local},\
                constraints = cons )
        currentMin = f( solutionLocal.x )
        feasible = check_feasibility(solutionLocal.x, bnds) 

        # if feasible == True: # jump from current point even if it is not optimum
        #     x = solutionLocal.x

        # Check te current point from which to jump: after doing a long jump or
        # when the solution is improved        
        if jumpMagnitude == 1 or currentMin < previousMin:
            x = solutionLocal.x
            
        # Check improvement      
        if currentMin < bestMin and feasible == True: # Improvement            
            Best = x
            bestMin = currentMin
            accepted = True

            # If the improvement is not large, assume it is not improvement
            if (previousMin - currentMin ) < tolGlobal: 
                n_noimprove += 1
            else:
                n_noimprove = 0
            jumpMagnitude = jumpMagnitude_default

        elif n_noimprove == niter_success: # Not much improvement
            accepted = False
            jumpMagnitude = 1
            n_noimprove = 0 # Restart count so that it performs jumps around a point
        else:
            accepted = False
            jumpMagnitude = jumpMagnitude_default
            n_noimprove += 1        

        previousMin = currentMin

        # Save results every 5 iter
        if n_itercounter % 20 == 0:
            AL_BF.writeData(Best, 'w', 'SolutionMBH_self.txt')
        
        print("iter", n_itercounter)
        print("Current min vs best one", currentMin, bestMin)
        print_fun(f, x, accepted)

    # Print solution 
    # print_sol(Best, bestMin, n_itercounter, niter_success)
    return Best, bestMin
        

###############################################
# EVOLUTIONARY ALGORITHM
###############################################
# Random initialization--> fitness calculation--> offspring creation --> immigration, mutation --> convergence criterion
def EvolAlgorithm(f, bounds, *args, **kwargs):
    """
    EvolAlgorithm: evolutionary algorithm
    INPUTS:
        f: function to be analyzed
        x: decision variables
        bounds: bounds of x to initialize the random function
        x_add: additional parameters for the function. As a vector
        ind: number of individuals. 
        cuts: number of cuts to the variable
        tol: tolerance for convergence
        max_iter: maximum number of iterations (generations)
        max_iter_success
        elitism: percentage of population elitism
        bulk_fitness: if True, the data has to be passed to the function all 
                    at once as a matrix with each row being an individual
    """
    x_add = kwargs.get('x_add', False)
    ind = kwargs.get('ind', 100)
    cuts = kwargs.get('cuts', 1)
    tol = kwargs.get('tol', 1e-4)
    max_iter = kwargs.get('max_iter', 1e2)
    max_iter_success = kwargs.get('max_iter_success', 1e1)
    elitism = kwargs.get('elitism',0.1)
    mut = kwargs.get('mutation',0.01)
    cons = kwargs.get('cons', None)
    bulk = kwargs.get('bulk_fitness', False)
    typex = kwargs.get('typex', False)

    path_save = kwargs.get('path_save', './')

    def f_evaluate(pop_0, x_add):
        if bulk == True:
            if x_add == False:
                pop_0[:,0] = f(pop_0[:,1:])
            else:
                pop_0[:,0] = f(pop_0[:,1:], x_add)
        else:
            if x_add == False: # No additional arguments needed
                for i in range(ind):
                    print("Individual: %i/%i"%(i, ind))
                    pop_0[i,0] = f(pop_0[i,1:])
            else:
                for i in range(ind):
                    x_add2 = x_add.copy()
                    x_add2.append(i)
                    print("Individual: %i/%i"%(i, ind))
                    pop_0[i,0] = f(pop_0[i,1:], x_add2)

        return pop_0

    ###############################################
    ###### GENERATION OF INITIAL POPULATION #######
    ###############################################
    pop_0 = np.zeros([ind, len(bounds)+1])
    for i in range(len(bounds)):
        if typex[i] == 'int':
            pop_0[:, i+1] = np.random.randint(low = bounds[i,0], high = bounds[i,1], size = ind, dtype = int)
        elif typex[i] == 'log':
            log_distrib = np.random.randint(low = np.log10(bounds[i,0]), high = np.log10(bounds[i,1]), size = ind, dtype = int)
            pop_0[:, i+1] = np.array([10.0**i for i in log_distrib])
        else:
            pop_0[:, i+1] = np.random.rand(ind) * (bounds[i, 1]-bounds[i, 0]) + bounds[i, 0]

    with open(path_save + "evol_population_initial.csv", "w+") as output:
        wr = csv.writer(output)
        wr.writerows(pop_0)

    ###############################################
    ###### FITNESS EVALUATION               #######
    ###############################################
    if x_add != False:
        x_add2 = x_add.copy()
        x_add2.append(0)
    pop_0 = f_evaluate(pop_0, x_add2)
    
    Sol = pop_0[pop_0[:,0].argsort()]
    minVal = min(Sol[:,0])
    x_minVal = Sol[0,:]

    # with open('population.txt', 'w') as myfile:
    #     myfile.write(pop_0)
    with open(path_save + "evol_population.csv", "w+") as output:
        wr = csv.writer(output)
        wr.writerows(pop_0)
    
    ###############################################
    ###### NEXT GENERATION                  #######
    ###############################################
    noImprove = 0
    counter = 0
    lastMin = minVal
    
    Best = np.zeros([max_iter+1,len(bounds)+1])
    while noImprove <= max_iter_success and counter <= max_iter :
        
        print("Iteration: %i/%i"%(counter, max_iter))
        ###############################################
        #Generate descendents

        #Elitism
        ind_elit = int(round(elitism*ind))

        children = np.zeros(np.shape(pop_0))
        children[:,1:] = Sol[:,1:]

        #Separate into the number of parents  
        pop = np.zeros(np.shape(pop_0))
        pop[:ind_elit,:] = children[:ind_elit,:] #Keep best ones
        np.random.shuffle(children[:,:]) #shuffle the others
        

        for j in range ( (len(children)-ind_elit) //2 ):
            if len(bounds) == 2:
                cut = 1
            else:
                cut = np.random.randint(1,len(bounds)-1)

            pop[ind_elit +2*j,1:] = np.concatenate((children[2*j,1:cut+1],children[2*j +1,cut+1:]),axis = 0)
            pop[ind_elit+ 2*j + 1,1:] = np.concatenate((children[2*j+1,1:cut+1],children[2*j ,cut+1:]),axis = 0)
        
        if (len(children)-ind_elit) %2 != 0:
            pop[-1,:] = children[-ind_elit,:]

        #Mutation
        for i in range(ind):
            for j in range(len(bounds)):
                if np.random.rand(1) < mut: #probability of mut                    
                    if typex[j] == 'int':
                        pop[i,j+1] = np.random.randint(low = bounds[j,0], high = bounds[j,1], size = 1, dtype = int)
                    else:
                        pop[i,j+1]= np.random.rand(1) * (bounds[j, 1]-bounds[j, 0]) + bounds[j, 0]

        ###############################################
        # Fitness
        if x_add != False:
            x_add2 = x_add.copy()
            x_add2.append(counter)
        pop = f_evaluate(pop, x_add2)

        Sol = pop[pop[:,0].argsort()]
        minVal = min(Sol[:,0])

        ###############################################
        #Check convergence        
        if  minVal >= lastMin: 
            noImprove += 1
#         elif abs(lastMin-minVal)/lastMin > tol:
#             noImprove += 1
        else:
#             print('here')
            lastMin = minVal 
            x_minVal = Sol[0,1:]
            noImprove = 0
            Best[counter,:] = Sol[0,:] 
        
        print(counter, "Minimum: ", minVal)
        counter += 1 #Count generations 

        # SAVE RESULTS
        # save all values
        # with open('population.txt', 'a') as myfile:
        #     myfile.write(pop)
        with open(path_save+ "evol_population.csv", "a") as output:
            wr = csv.writer(output)
            wr.writerows(pop)

        # save minimum
        if counter % 5 == 0:
            # AL_BF.writeData(x_minVal, 'w', 'SolutionEA.txt')
            str_data = ' '.join(str(i) for i in x_minVal)
            with open(path_save +'SolutionEA.txt', 'w') as fo:
                fo.write(str_data)
                fo.write("\n")
        
        # print(counter)
    print("minimum:", lastMin)
    print("Iterations:", counter)
    print("Iterations with no improvement:", noImprove)
    
    return x_minVal, lastMin
