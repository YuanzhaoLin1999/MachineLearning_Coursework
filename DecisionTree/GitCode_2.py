import pandas as pd
from decisionTree import *
from fileIO import *
from classifier import *
import random

 
class ID3:    
    def __init__(self,argvList):
        self.trainingData=fileIO(argvList[3]);
        self.examples=self.trainingData.masterCopy;
        self.DT=None;
        self.bestTree=self.DT;
        self.argvList=argvList;
    
    ##############################
    #The actual ID3 algorithm runs here.
    #1.Grow tree
    #2.Prune tree
    #input-String-Type of heuristic
    #returns-Nothing    
    ##############################    
    def runID3(self,heuristic):
        print("Growing tree...")
        self.growTree(self.trainingData.masterCopy,self.trainingData.allFeatures,"","",heuristic);
        print("Pruning Tree..");
        self.bestTree,bestAccuracy=self.pruneTree(int(self.argvList[1]),int(self.argvList[2]));
        print("validating the test data");
        if(self.argvList[6].lower()=="yes"):
            self.bestTree.printInorder(self.bestTree.root);
    ###################################
    #validates test data with the best tree
    #chosen after pruninig.
    #Input-None
    #output-the accuracy in percent
    ####################################
    def validateTestDataOrg(self):
        testData=fileIO(self.argvList[5]);
        return(self.DT.validate(testData));
    
    ###################################
    #validates test data with the best tree
    #chosen after pruninig.
    #Input-None
    #output-the accuracy in percent
    ####################################
    def validateTestData(self):
        testData=fileIO(self.argvList[5]);
        return(self.bestTree.validate(testData));       
            
    #####################################
    #the iterative ID3 algorithm
    #first stop condition-checks if all the samples are 
    #either completely positive or completely negative.
    #second stop condition-checks if there are rows 
    #containng that feature value in the current dataframe
    #third stop condition-all attributes are over
    #inputs-dataframe, remaining features, parentNode of the future 
    #chosen attribute, branchName of the fture node,the heuristic that was requested
    #output-None
    #####################################
    def growTree(self,dataFrame,remainingFeatures,parentNode,branchName,heurisitic):        
        for uniqueClassValue in self.trainingData.uniqueValDictionary[self.trainingData.targetFeature]:
            df=dataFrame.loc[dataFrame[self.trainingData.targetFeature]==uniqueClassValue];
            if(df.shape[0]==dataFrame.shape[0]):
                self.DT.addLeafNode(parentNode,branchName,uniqueClassValue);
                return;
        chosenFeature=chooseBestAttribute(dataFrame,self.trainingData,remainingFeatures,heurisitic);
        if(self.DT is None):
            self.DT=decisionTree(chosenFeature);
            nextParentNode=self.DT.root;
            self.DT.root.dataframe=dataFrame;
        else:
            nextParentNode=self.DT.addNonLeafNode(parentNode,chosenFeature,branchName,dataFrame);      
        
        for uniqueFeatureValue in self.trainingData.uniqueValDictionary[chosenFeature]:
            dfSubset=dataFrame.loc[dataFrame[chosenFeature]==uniqueFeatureValue];
            if(dfSubset.empty):
                if(uniqueFeatureValue==1):
                    self.DT.addLeafNode(nextParentNode,uniqueFeatureValue,'0');
                else:
                    self.DT.addLeafNode(nextParentNode,uniqueFeatureValue,'1');
            else:
                if(chosenFeature in remainingFeatures):
                    remFeatures=list(remainingFeatures);
                    remFeatures.remove(chosenFeature);
                    if(len(remFeatures)==1):
                        #print("all atrributes over");
                        majClassVal=self.findMajorityClassVal(dfSubset,self.trainingData.targetFeature);
                        self.DT.addLeafNode(nextParentNode,uniqueFeatureValue,majClassVal);
                        return;               
                self.growTree(dfSubset,remFeatures,nextParentNode,uniqueFeatureValue,heurisitic);
        return;
        
    #####################################
    #Finds and returns the majority classValue in the 
    #given dataframe
    #inputs-dataframe,class Feature Name
    #returns-the class Value that occurs in majority in the dataframe
    ####################################
    def findMajorityClassVal(self,df,tf):
        for uniqueClassVal in self.trainingData.uniqueValDictionary[tf]:
            majorityCount=0;
            majorityClassVal="";
            dfSubset=df.loc[df[tf]==uniqueClassVal];
            if dfSubset.shape[0]>majorityCount:
                majorityCount=dfSubset.shape[0];
                majorityClassVal=uniqueClassVal;
        return majorityClassVal;
        
        
        
    #############################
    #prunes the tree
    #input-L.K
    #returns the best tree and best accuracy.
    ##############################    
    def pruneTree(self,L,K):
        if(self.DT==None):
            raise SystemExit("Tree not built. This tree cannot be pruned");
        else:
            bestTree=self.DT.copyTree() ;
            validatingData=fileIO(self.argvList[4]);
            bestAccuracy=bestTree.validate(validatingData);
            #print("original Tree accuracy",bestAccuracy);
            for i in range(L):
                D2=self.DT.copyTree();
                M=random.randint(1,K);
                for j in range(M):
                    numOfNodes=D2.numberNodesOfTree();
                    if(numOfNodes==1 or numOfNodes== 0):
                        break;
                    else:
                        P=random.randint(1,numOfNodes);
                        D2.replaceSubtreeWithLeaf(P,self.trainingData);
                    #check for accuracy
                thisTreeAccuracy=D2.validate(validatingData);
                #print("this tree accuracy",thisTreeAccuracy);
                if(thisTreeAccuracy>bestAccuracy):
                    bestAccuracy=thisTreeAccuracy;
                    bestTree=D2.copyTree(); 
            #print("best Accuracy",bestAccuracy);
            return bestTree,bestAccuracy;