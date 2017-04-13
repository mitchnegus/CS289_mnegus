"""
randomforest.py
==============================================
Train a random forest and predict on test data
==============================================

This python module contains the random forest class, which can be trained on 
labeled data (input as numpy arrays: for data, an Nxd matrix with N rows 
corresponding to N sample points and d columns corresponding to d features; for 
labels, an N vector with labels corresponding to each of the N sample points)
"""

import numpy as np


class RandomDecisionTree:
    """
    Build and store a random decision tree, based on supplied training data. 
    Use this tree to predict classifications.
    - treedepth: an integer for the max depth of the tree
    - mfeatures: an integer number of random features tested for splits per node
    - verbose:   a boolean for descriptive output
    """

    def __init__(self,treedepth=10,mfeatures=None,verbose=False):
        self.depth = treedepth
        self.mfeatures = mfeatures
        self.nfeatures = None
        self.verbose = verbose
        self.tree = self.Node()
        if type(treedepth) is not int:
            print('ERROR (RandomDecisionTree): Tree depth must be an integer.')
        if mfeatures and type(mfeatures) is not int:
            print('ERROR (RandomDecisionTree): The number of random features must be an integer.')
        
        
    def entropy(self,C,D,c,d):
        """
        Calculate entropy based on classifications above and below the 
        splitrule.
        - C: sample points in-class below splitrule (left)
        - D: sample points not-in-class below splitrule (left)
        - c: sample points in-class above splitrule (right)
        - d: sample points not-in-class above splitrule (right)
        Returns the entropy.
        """
        
        if C != 0:
            Cfactor = -(C/(C+D))*np.log2(C/(C+D))
        else:
            Cfactor = 0
        if D != 0:
            Dfactor = -(D/(C+D))*np.log2(D/(C+D))
        else:
            Dfactor = 0
        if c != 0:
            cfactor = -(c/(c+d))*np.log2(c/(c+d))
        else:
            cfactor = 0
        if d != 0:
            dfactor = -(d/(c+d))*np.log2(d/(c+d))
        else:
            dfactor = 0
        H_left = Cfactor + Dfactor
        H_right = cfactor + dfactor
        H = ((C+D)*H_left + (c+d)*H_right)/(C+D+c+d)
        
        return H
    
    
    def pick_random_features(self):
        """Randomly choose a set of m features out of n total features"""
        
        mrandomfeatures = -1*np.ones(self.mfeatures)
        for i in range(self.mfeatures):
            while mrandomfeatures[i] == -1:
                feature_i = np.random.randint(self.nfeatures)
                if feature_i not in mrandomfeatures:
                    mrandomfeatures[i] = feature_i
        mrandomfeatures = np.sort(mrandomfeatures).astype(int)
        
        return mrandomfeatures 
    
    
    def segment(self,data,labels):
        """
        March through data and determine split which maximizes info gain.
        Returns the ideal splitrule as a length-2 list where the first element
        is the index of the splitting feature and the second element is the 
        value of that feature to split on.
        """
        
        totals = np.bincount(labels)
        if len(totals)==1:
            totals = np.append(totals,[0])
        # Quick safety check
        if len(labels) != len(data):
            print('ERROR (RandomForest.segment): There must be the same number of labels as datapoints.')
        
        # Calculate the initial entropy, used to find info gain
        C,D = 0,0                      # C = in class left of split; D = not in class left of split
        c,d = totals[1],totals[0]      # c = in class right of split; d = not in class right of split
        H_i = self.entropy(C,D,c,d) # the initial entropy, before any splitting
        
        # Initialize objects to store optimal split rules for iterative comparison
        maxinfogain = 0
        splitrule = []   
        
        mrandomfeatures = self.pick_random_features()
        for feature_i in mrandomfeatures:
            # Order the data for determining ideal splits
            lbldat = np.concatenate(([data[:,feature_i]],[labels]),axis=0)
            
            fv = np.sort(lbldat.T,axis=0)
            lastfeature = np.array(['',''])
            
            C,D = 0,0                      # Reset the counters
            c,d = totals[1],totals[0]
            
            for point_i in range(len(fv)-1):
                
                # Update C,D,c,d to minmize runtime of entrop calc (keep at O(1) time)
                if fv[point_i,1] == 1:
                    C += 1
                    c -= 1
                elif fv[point_i,1] == 0:
                    D += 1
                    d -= 1
                else:
                    print("ERROR (RandomForest.segment): Classifications can only be 0 or 1.")
                
                # Skip splitting values that are not separable
                if fv[point_i,0] == fv[point_i+1,0]:
                    continue
                else:
                    H_f = self.entropy(C,D,c,d)
                    infogain = H_i-H_f
                    if infogain > maxinfogain:
                        maxinfogain = infogain
                        splitrule = [feature_i,fv[point_i,0]]
        
        return splitrule
        
        
    def train(self,data,labels,node=1,deep=0):
        """
        Train the random decision tree on input data
        - data:   Nxd numppy array with N sample points and d features
        - labels: 1D, length-N numpy array with labels for the N sample points
        - node:   node class passed to function; default is 1, a flag for the head node (INTERNAL USE ONLY)
        - deep:   a counter to determine current depth in tree (INTERNAL USE ONLY)
        """
        
        # Ensure labels are integers
        labels = labels.astype(int)

        # On the first training cycle, set the current node to the head node
        # If the number of random features has not yet been set, set that too.
        if node==1:
            node=self.tree
            if self.mfeatures is None:
                self.mfeatures = np.int(np.round((np.sqrt(len(data[0])))))  # m random features
            self.nfeatures = len(data[0])                    # n total features
            if self.mfeatures > self.nfeatures:
                print('WARNING: The number of random features to choose is greater than the total number of features. Using all of the features instead')
                self.mfeatures = self.nfeatures
        # Grow decision tree
        depthlim = self.depth
        if deep < depthlim:
            splitrule = self.segment(data,labels)
        else:
            splitrule = []
        if self.verbose is True:
            print(data,labels)
        node.isleaf(data,labels,splitrule)
        
        # Train deeper if the node splits
        if node.nodetype == 'SplitNode':
            if self.verbose is True:
                print('rule:',node.rule)
                print('Splitting node left and right')
            deep += 1
            node.left=self.Node()
            node.right=self.Node()
            self.train(node.leftdata,node.leftlabels,node.left,deep)
            self.train(node.rightdata,node.rightlabels,node.right,deep)
        elif node.nodetype == 'LeafNode':
            if self.verbose is True:
                print('You made a leaf node! It has value',node.leaflabel,'and',node.leafcount,'items.')            
        else:
            print('ERROR (RandomForest.train): The node type could not be identified!')
    
        
    def predict(self,testdata):
        """
        Predict classfications for unlabeled data points using the previously 
        trained random decision tree.
        - testdata: Nxd numpy array with N sample points and d features
                    *Note, dimensions N and d must match those used 
                    for data array in RandomDecisionTree.train*
        Returns a 1D, length-N numpy array of predictions (one prediction per point)
        """
        
        npoints = len(testdata)
        predictions = np.empty(npoints)
        for point_i in range(npoints):
            ParentNode = self.tree
            Rule = ParentNode.rule
            while Rule is not None:
                splitfeat_i = Rule[0]
                splitval = Rule[1]
                if testdata[point_i,splitfeat_i] <= splitval:
                    ChildNode = ParentNode.left
                else:
                    ChildNode = ParentNode.right
                ParentNode = ChildNode
                Rule = ParentNode.rule
            predictions[point_i]=ParentNode.leaflabel
        
        return predictions.astype(int)
    
    
    
    class Node:
        """
        Store a decision tree node, coupled in series to construct tree;
        includes a left branch, right branch, and splitrule
        """
    
        def __init__(self):
            self.rule = None
            self.left = None
            self.leftdata = None
            self.leftlabels = None
            self.right = None
            self.rightdata = None
            self.rightlabels = None
            self.leaflabel = None
            self.leafcount = None
            self.nodetype = None
            
            
        def isleaf(self,data,labels,splitrule):
            """Determine if this is a leaf node"""
            
            if splitrule:
                indsabove = self.datainds_above_split(data,splitrule)
                self.rule = splitrule
                self.leftdata,self.leftlabels = self.leftDL(data,labels,indsabove)
                self.rightdata,self.rightlabels = self.rightDL(data,labels,indsabove)
                self.nodetype = 'SplitNode'
            
            elif not splitrule:
                self.leaflabel = np.bincount(labels).argmax()
                self.leafcount = len(labels)
                self.nodetype = 'LeafNode'

         
        def datainds_above_split(self,data,splitrule):
            """
            Collect indices of points with values of the splitting feature
            greater than the split rule
            """
            
            indsabove = []
            fv = data[:,splitrule[0]]
            for point_i in range(len(fv)):
                if fv[point_i] > splitrule[1]:
                    indsabove.append(point_i)
            
            return indsabove
              
        
        def leftDL(self,data,labels,indsabove):
            """Return arrays of only left data and labels"""
            
            leftdata = np.delete(data,indsabove,axis=0)
            leftlabels = np.delete(labels,indsabove,axis=0)
            
            return leftdata,leftlabels   
        
        
        def rightDL(self,data,labels,indsabove):
            """Return arrays of only right data and labels"""
            
            rightdata = data[indsabove]
            rightlabels = labels[indsabove]
            
            return rightdata,rightlabels
        

class RandomForest:
    """
    Build and store a random forest, based on supplied training data. 
    Use this tree to predict classifications.
    - treedepth: an integer for the max depth of any tree in the forest
    - mfeatures: an integer number of random features tested for splits per node
    - verbose:   a boolean for descriptive output
    """

    def __init__(self,treedepth=10,ntrees=None,mfeatures=None,subsize=None,verbose=False):
        self.treedepth = treedepth
        self.mfeatures = mfeatures
        self.subsize = subsize
        self.verbose = verbose
        self.treecount = ntrees
        self.forest = []
        if type(treedepth) is not int:
            print('ERROR (RandomForest): Tree depth must be an integer.')
        if mfeatures and type(mfeatures) is not int:
            print('ERROR (RandomForest): The number of random features must be an integer.')
            
            
    def train(self,data,labels):
        """
        Train (grow) the random forest on input data
        - data:   Nxd numppy array with N sample points and d features
        - labels: 1D, length-N numpy array with labels for the N sample points
        """
        if self.subsize is None:
            self.subsize = len(data)
        if self.treecount is None:
            self.treecount = int(np.sqrt(len(data)))
        elif type(self.treecount) is not int:
            print('ERROR (RandomForest): The number of trees must be an integer.')
            
        for tree_i in range(self.treecount):
            # choose a random subset of the data, size "subsize", for BAGGING
            subsetindices = np.random.randint(0,self.subsize,self.subsize)
            baggeddata = data[subsetindices]
            baggedlabels = labels[subsetindices]
            tree = RandomDecisionTree(self.treedepth,self.mfeatures,self.verbose)
            tree.train(baggeddata,baggedlabels)
            self.forest.append(tree)
            if tree_i%5 == 0:
                print('Finished training %i tree(s) out of %i' %(tree_i,self.treecount))
            
    def predict(self,testdata):
        """
        Predict classfications for unlabeled data points using the previously 
        trained random forest.
        - testdata: Nxd numpy array with N sample points and d features
                    *Note, dimensions N and d must match those used 
                    for data array in RandomForest.train*
        Returns a 1D, length-N numpy array of predictions (one prediction per point)
        """
        
        aggregatedpredictions = np.empty((self.treecount,len(testdata)))
        for tree_i in range(self.treecount):
            treepredictions = self.forest[tree_i].predict(testdata)
            aggregatedpredictions[tree_i]=treepredictions
        forestpredictions = np.round(np.average(aggregatedpredictions,axis=0)).astype(int)
        
        return forestpredictions
    