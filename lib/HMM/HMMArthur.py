#coding: utf-8
import numpy as np

class HMM:
    def __init__(self, Ann, Bnm, pi1n):
        self.A = np.array(Ann)
        self.B = np.array(Bnm)
        self.pi = np.array(pi1n)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]
        
    def printhmm(self):
        print "=================================================="
        print "HMM content: N =",self.N,",M =",self.M
        print 'hmm.pi'
        print self.pi
        print 'hmm.A'
        print self.A
        print 'hmm.B'
        print self.B
        print "=================================================="

    # Forward: 前向算法alpha  # O:观察值序列 # alpha: 前向概率 # pprob: 最终生成概率
    def Forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.N))
        # 递归计算alpha
        alpha[0, :] = self.pi * self.B[:, O[0]]
        for t in range(T-1):
            alpha[t+1, :] = np.dot(alpha[t, :], self.A) * self.B[:, O[t+1]]
        # 最终概率
        pprob = np.sum(alpha[T-1, :])
        # 返回
        return (alpha, pprob)
    
    # 带修正的前向算法
    def ForwardWithScale(self, O):
        T = len(O)
        alpha = np.zeros((T, self.N))
        scale = np.zeros(T)
        # 递推
        alpha[0, :] = self.pi * self.B[:, O[0]]
        scale[0] = sum(alpha[0, :])
        alpha[0, :] /= scale[0]
        for t in range(T-1):
            alpha[t+1, :] = np.dot(alpha[t, :], self.A) * self.B[:, O[t+1]]
            scale[t+1] = sum(alpha[t+1, :])
            alpha[t+1, :] /= scale[t+1]
        # 概率: 用于学习时看概率是否还有变化
        pprob = np.sum(np.log(scale))
        # 返回
        return (alpha, pprob)

    # Backward: 后向计算beta # O:观察值序列 
    def Backward(self, O):
        T = len(O)
        beta = np.zeros((T, self.N))
        # 递推
        beta[T-1, :] = 1.0
        for t in range(T-2,-1,-1):
            beta[t, :] = np.dot(self.A, self.B[:, O[t+1]] * beta[t+1, :])
        # return
        return beta

    # 带修正的后向算法
    def BackwardWithScale(self, O):
        T = len(O)
        beta = np.zeros((T, self.N))
        # 递推
        beta[T-1, :] = 1.0
        for t in range(T-2,-1,-1):
            beta[t, :] = np.dot(self.A, self.B[:, O[t+1]] * beta[t+1, :])
            beta[t, :] /= sum(beta[t, :])
        # return
        return beta
    
    # Viterbi算法: Direction(最大概率路径), prob(最大概率)
    def viterbi(self,O):
        T = len(O)
        delta = np.zeros((T,self.N),np.float)  
        MaxIndex = np.zeros((T,self.N),np.float)  
        Direction = np.zeros(T)
        # 递推
        delta[0, :] = self.pi * self.B[:,O[0]]
        for t in range(1,T):
            deltaA = (delta[t-1, :] * self.A.T).T
            delta[t, :] = deltaA.max(axis=0) * self.B[:, O[t]]
            MaxIndex[t, :] = deltaA.argmax(axis=0)
        # 倒序获取状态序列
        prob = delta[T-1, :].max()
        Direction[T-1] = delta[T-1, :].argmax()
        for t in range(T-2,-1,-1): 
            Direction[t] = MaxIndex[t+1, Direction[t+1]]
        # return
        return Direction, prob
    
    # 计算gamma: 时刻t时马尔可夫链处于状态i的概率
    def ComputeGamma(self, alpha, beta):
        gamma = alpha * beta
        gamma = (gamma.T / np.sum(gamma, axis=1)).T
        return gamma
    
    # 计算xi
    def ComputeXi(self, O, alpha, beta):
        T = len(O)
        xi = np.zeros((T, self.N, self.N))
        for t in range(T-1):
            xi[t, :, :] = (alpha[t, :] * self.A.T).T * (beta[t+1, :] * self.B[:, O[t+1]])
            xi[t, :, :] /= np.sum(xi[t, :, :])
        return xi

    # Baum-Welch算法:O为L个观察序列，每个观察序列可以一不样大小
    def BW(self, O):
        print "BaumWelch"
        DELTA = 0.01 ; round = 0 ; flag = 1 ; probf = 0.0
        delta = 0.0 ; deltaprev = 0.0 ; probprev = 0.0 ; ratio = 0.0 ; deltaprev = 10e-70
        
        while True :
            probf = 0
            pi = np.zeros(self.N)
            denominatorA = np.zeros((self.N),np.float)
            denominatorB = np.zeros((self.N),np.float)
            numeratorA = np.zeros((self.N,self.N),np.float)
            numeratorB = np.zeros((self.N,self.M),np.float)

            # E - step
            L = len(O)
            for index in range(L):
                Ob = np.array(O[index])
                T = len(O[index])

                (alpha, pprob) = self.ForwardWithScale(Ob)
                beta = self.BackwardWithScale(Ob)
                gamma = self.ComputeGamma(alpha, beta)
                xi = self.ComputeXi(Ob, alpha, beta)

                probf += pprob
                pi += gamma[0, :]

                denominatorA += np.sum(gamma[0:T-1, :], axis=0)
                denominatorB += np.sum(gamma, axis=0)
                numeratorA += np.sum(xi[0:T-1, :, :], axis=0)
                for k in range(self.M):
                    validFlag = Ob==k
                    validGamma = (gamma.T*validFlag).T
                    numeratorB[:, k] += np.sum(validGamma, axis=0)

            # 重新估计pi, A, B
            self.pi = 0.001/self.N + 0.999*pi/L
            self.A = 0.001/self.N + 0.999*(numeratorA.T / denominatorA).T
            self.B = 0.001/self.M + 0.999*(numeratorB.T / denominatorB).T

            if flag == 1:
                flag = 0
                probprev = probf
                ratio = 1
                continue
            delta = probf - probprev
            ratio = delta / deltaprev
            probprev = probf
            deltaprev = delta
            round += 1
            print 'iteration: %d' %(round)
            if ratio <= DELTA :
                break
         

if __name__ == "__main__":
    A = [[0.8125,0.1875],[0.2,0.8]]
    B = [[0.875,0.125],[0.25,0.75]]
    pi = [0.5,0.5]
    hmm = HMM(A, B, pi)
    
    O = [[1,0,0,1,1,0,0,0,0],
         [1,1,0,1,0,0,1,1,0],
         [0,0,1,1,0,0,1,1,1]]

    hmm.BW(O)
    hmm.printhmm()




















