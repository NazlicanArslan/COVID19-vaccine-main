'''
Epidemiology parameters from TACC simulation model
Author of the epi model: Zhanwei Du

Texas Advanced Computing Center (TACC)
'''
import numpy as np
from itertools import product


class EpiSetup:
    '''
        A setup for the epidemiological parameters.
        Scenarios 6 corresponds to best guess parameters for UT group.
    '''
    def __init__(self):
        '''
            Initialize an instance of epidemiological parameters. If the
            parameter is random, is not initialize and is queried as a
            property
        '''
        # Contact matrices
        self.phi_all = np.array([
            [2.1600, 2.1600, 4.1200, 0.8090, 0.2810],
            [0.5970, 8.1500, 5.4100, 0.7370, 0.2260],
            [0.3820, 2.4300, 10.2000, 1.7000, 0.2100],
            [0.3520, 1.8900, 6.7100, 3.0600, 0.5000],
            [0.1900, 0.8930, 2.3900, 1.2000, 1.2300],
        ])
        
        self.phi_school = np.array([
            [0.9950, 0.4920, 0.3830, 0.0582, 0.0015],
            [0.1680, 3.7200, 0.9260, 0.0879, 0.0025],
            [0.0428, 0.6750, 0.8060, 0.0456, 0.0026],
            [0.0842, 0.7850, 0.4580, 0.0784, 0.0059],
            [0.0063, 0.0425, 0.0512, 0.0353, 0.0254],
        ])
        
        self.phi_work = np.array([
            [0, 0, 0, 0, 0.0000121],
            [0, 0.0787, 0.4340000, 0.0499, 0.0003990],
            [0, 0.181, 4.490, 0.842, 0.00772],
            [0, 0.131, 2.780, 0.889, 0.00731],
            [0.00000261, 0.0034900, 0.0706000, 0.0247, 0.0002830],
        ])
    
    @classmethod
    def load_file(cls, params):
        epi_params = cls()
        for (k, v) in params.items():
            if isinstance(v, list):
                if v[0] == "rnd_inverse" or v[0] == "rnd":
                    setattr(epi_params, k, ParamDistribution(*v))
                else:
                    setattr(epi_params, k, np.array(v))
            else:
                setattr(epi_params, k, v)
        return epi_params
    
    def update_rnd_stream(self, rnd_stream):
        '''
            Generates random parametes from a given random stream.
            Coupled parameters are updated as well.
            Args:
                rnd_stream (RandomState): a RandomState instance from numpy.
        '''

        #rnd_stream = None  #rnd_stream
        tempRecord = {}
        for k in vars(self):
            v = getattr(self, k)
            # if the attribute is random variable, generate a deterministic version
            if isinstance(v, ParamDistribution):
                tempRecord[v.param_name] = v.sample(rnd_stream)
            elif isinstance(v, np.ndarray):
                listDistrn = True
                # if it is a list of random variable, generate a list of deterministic values
                vList = []
                outList = []
                outName = None
                for vItem in v:
                    try:
                        vValue = ParamDistribution(*vItem)
                        outList.append(vValue.sample(rnd_stream))
                        outName = vValue.param_name
                    except:
                        vValue = 0
                    vList.append(vValue)
                    listDistrn = listDistrn and isinstance(vValue, ParamDistribution)
                if listDistrn:
                    tempRecord[outName] = np.array(outList)

        # print(tempRecord)

        for k in tempRecord.keys():
            setattr(self, k, tempRecord[k])
            
        # self._omega_E = np.array([((YHR[a] / self.Eta[a]) +
        #                            ((1 - YHR[a]) / self._gamma_IY[a])) * self.omega_IY * self._sigma_E * self.pp /
        #                           (1 - self.pp) for a in range(len(YHR))])
        # omega is computed with overall hosp rate
        self.omega_P = np.array([(self.tau * self.omega_IY * (self.YHR_overall[a] / self.Eta[a] +
                                                              (1 - self.YHR_overall[a]) / self.gamma_IY) +
                                  (1 - self.tau) * self.omega_IA / self.gamma_IA) /
                                 (self.tau * self.omega_IY +
                                  (1 - self.tau) * self.omega_IA) * self.rho_Y * self.pp / (1 - self.pp)
                                 for a in range(len(self.YHR_overall))])
        self.omega_PA = self.omega_IA * self.omega_P
        self.omega_PY = self.omega_IY * self.omega_P
        # self.omega_PA = np.array([0.91117513, 0.91117513, 0.924606534, 0.957988874, 0.98451149])
        # self.omega_PY = np.array([1.366762694, 1.366762694, 1.386909802, 1.436983311, 1.476767236])
        # pi is computed using risk based hosp rate
        self.pi = np.array([
            self.YHR[a] * self.gamma_IY / (self.Eta[a] + (self.gamma_IY - self.Eta[a]) * self.YHR[a])
            for a in range(len(self.YHR))
        ])
        self.YFR = self.IFR / self.tau
        self.HFR = self.YFR / self.YHR
        self.rIH0 = self.rIH
        self.YHR0 = self.YHR
        self.YHR_overall0 = self.YHR_overall
        # if gamma_IH and mu are lists, reshape them for right dimension
        if isinstance(self.gamma_IH,np.ndarray):
            self.gamma_IH = self.gamma_IH.reshape(self.gamma_IH.size,1)
            self.gamma_IH0 = self.gamma_IH.copy()
        if isinstance(self.mu,np.ndarray):
            self.mu = self.mu.reshape(self.mu.size,1)
            self.mu0 = self.mu.copy()
        try:
            self.HICUR0 = self.HICUR
            self.nu = self.gamma_IH * self.HICUR / (self.mu + (self.gamma_IH- self.mu) * self.HICUR)
            if isinstance(self.gamma_ICU,np.ndarray):
                self.gamma_ICU = self.gamma_ICU.reshape(self.gamma_ICU.size,1)
                self.gamma_ICU0 = self.gamma_ICU.copy()
            if isinstance(self.mu_ICU,np.ndarray):
                self.mu_ICU = self.mu_ICU.reshape(self.mu_ICU.size,1)
                self.mu_ICU0 = self.mu_ICU.copy()
            self.nu_ICU = self.gamma_ICU * self.ICUFR / (self.mu_ICU + (self.gamma_ICU- self.mu_ICU) * self.ICUFR)
        except:
            self.nu = self.gamma_IH * self.HFR / (self.mu + (self.gamma_IH- self.mu) * self.HFR)

        self.beta = self.beta0
        
    def delta_update_param(self, prev):
        '''
            Update parameters according to delta variant prevelance.
        '''
        
        self.beta = self.beta0 * (1 - prev) + self.beta0 * (1.65) * prev #increased transmission
             
        E_new = 1 / self.sigma_E - 1.5
        self.sigma_E = self.sigma_E * (1 - prev) + (1/ E_new) * prev #decreased incubation period.
       
        self.YHR = self.YHR * (1 - prev) + self.YHR * (1.8) * prev #increased hospitalization rate.
        self.YHR_overall = self.YHR_overall * (1 - prev) + self.YHR_overall * (1.8) * prev 
        
        #Update parameters where YHR is used:
        self.omega_P = np.array([(self.tau * self.omega_IY * (self.YHR_overall[a] / self.Eta[a] +
                                                              (1 - self.YHR_overall[a]) / self.gamma_IY) +
                                  (1 - self.tau) * self.omega_IA / self.gamma_IA) /
                                  (self.tau * self.omega_IY +
                                  (1 - self.tau) * self.omega_IA) * self.rho_Y * self.pp / (1 - self.pp)
                                  for a in range(len(self.YHR_overall))])
        self.omega_PA = self.omega_IA * self.omega_P
        self.omega_PY = self.omega_IY * self.omega_P
              
        
        self.pi = np.array([
            self.YHR[a] * self.gamma_IY / (self.Eta[a] + (self.gamma_IY - self.Eta[a]) * self.YHR[a])
            for a in range(len(self.YHR))
        ])
        self.HFR = self.YFR / self.YHR
        
        try:
            self.HICUR0 = self.HICUR
            self.nu = self.gamma_IH * self.HICUR / (self.mu + (self.gamma_IH- self.mu) * self.HICUR)
            if isinstance(self.gamma_ICU,np.ndarray):
                self.gamma_ICU = self.gamma_ICU.reshape(self.gamma_ICU.size,1)
                self.gamma_ICU0 = self.gamma_ICU.copy()
            if isinstance(self.mu_ICU,np.ndarray):
                self.mu_ICU = self.mu_ICU.reshape(self.mu_ICU.size,1)
                self.mu_ICU0 = self.mu_ICU.copy()
            self.nu_ICU = self.gamma_ICU * self.ICUFR / (self.mu_ICU + (self.gamma_ICU- self.mu_ICU) * self.ICUFR)
        except:
            self.nu = self.gamma_IH * self.HFR / (self.mu + (self.gamma_IH - self.mu) * self.HFR)

        
        # Update hospital dynamic parameters:
        self.gamma_ICU = (self.gamma_ICU0*(1 + self.alpha1)) * (1 - prev) + (self.gamma_ICU0 * 0.65 *(1 + self.alpha1_delta)) * prev
        self.mu_ICU = (self.mu_ICU0*(1 + self.alpha3)) * (1 - prev) + (self.mu_ICU0 * 0.65 * (1 + self.alpha3_delta)) * prev
        self.gamma_IH = (self.gamma_IH0*(1 - self.alpha2)) * (1 - prev) + (self.gamma_IH0 * (1 - self.alpha2_delta)) * prev
       
        self.alpha4 = self.alpha4_delta * prev + self.alpha4 * (1 - prev)
        
    def omicron_update_param(self, prev):
        '''
            Update parameters according omicron. 
            Assume increase in the tranmission.
            The changes in hosp dynamic in Austin right before omicron emerged.
        '''       
        self.beta = self.beta * (1 - prev) + self.beta * (self.omicron_beta) * prev #increased transmission
    
        self.YHR = self.YHR0 * (1 - prev) + self.YHR0 * 0.9 * prev
        self.YHR_overall = self.YHR_overall * (1 - prev) + self.YHR_overall * 0.9 * prev
        
        #Update parameters where YHR is used:
        self.omega_P = np.array([(self.tau * self.omega_IY * (self.YHR_overall[a] / self.Eta[a] +
                                                              (1 - self.YHR_overall[a]) / self.gamma_IY) +
                                  (1 - self.tau) * self.omega_IA / self.gamma_IA) /
                                  (self.tau * self.omega_IY +
                                  (1 - self.tau) * self.omega_IA) * self.rho_Y * self.pp / (1 - self.pp)
                                  for a in range(len(self.YHR_overall))])
        self.omega_PA = self.omega_IA * self.omega_P
        self.omega_PY = self.omega_IY * self.omega_P

        self.pi = np.array([
            self.YHR[a] * self.gamma_IY / (self.Eta[a] + (self.gamma_IY - self.Eta[a]) * self.YHR[a])
            for a in range(len(self.YHR))
        ])
        self.HFR = self.YFR / self.YHR
        
        try:
            self.HICUR0 = self.HICUR
            self.nu = self.gamma_IH * self.HICUR / (self.mu + (self.gamma_IH- self.mu) * self.HICUR)
            if isinstance(self.gamma_ICU,np.ndarray):
                self.gamma_ICU = self.gamma_ICU.reshape(self.gamma_ICU.size,1)
                self.gamma_ICU0 = self.gamma_ICU.copy()
            if isinstance(self.mu_ICU,np.ndarray):
                self.mu_ICU = self.mu_ICU.reshape(self.mu_ICU.size,1)
                self.mu_ICU0 = self.mu_ICU.copy()
            self.nu_ICU = self.gamma_ICU * self.ICUFR / (self.mu_ICU + (self.gamma_ICU- self.mu_ICU) * self.ICUFR)
        except:
            self.nu = self.gamma_IH * self.HFR / (self.mu + (self.gamma_IH- self.mu) * self.HFR)

         # Update hospital dynamic parameters:
        self.gamma_ICU = self.gamma_ICU0 *(1 + self.alpha1_omic) * 1.1 * prev + (self.gamma_ICU0 * 0.65 *(1 + self.alpha1_delta)) * (1 - prev)
        self.mu_ICU = self.mu_ICU0 * (1 + self.alpha3_omic) * prev + (self.mu_ICU0 * 0.65 * (1 + self.alpha3_delta)) * (1 - prev)
        self.gamma_IH = self.gamma_IH0 * (1 - self.alpha2_omic) * prev + (self.gamma_IH0 * (1 - self.alpha2_delta)) * (1 - prev)
        
        self.alpha4 = self.alpha4_omic * prev + self.alpha4_delta * (1 - prev)
        
    def variant_update_param(self, prev):
        '''
            Assume an imaginary new variant that is more transmissible.
        '''       
        self.beta = self.beta * (1 - prev) + self.beta * (self.new_variant_beta) * prev #increased transmission
     
       
    @property
    def eq_mu(self):
        # A conservative estimation of hospital service rate
        # Side computation for square root staffing
        self.update_rnd_stream(None)
        return np.minimum(self.gamma_IH, self.mu)
    
    def effective_phi(self, school, cocooning, social_distance, demographics, day_type):
        '''
            school (int): yes (1) / no (0) schools are closed
            cocooning (float): percentage of transmition reduction [0,1]
            social_distance (int): percentage of social distance (0,1)
            demographics (ndarray): demographics by age and risk group
            day_type (int): 1 Weekday, 2 Weekend, 3 Holiday, 4 Long Holiday
        '''
        
        A = len(demographics)  # number of age groups
        L = len(demographics[0])  # number of risk groups
        d = demographics  # A x L demographic data
        phi_all_extended = np.zeros((A, L, A, L))
        phi_school_extended = np.zeros((A, L, A, L))
        phi_work_extended = np.zeros((A, L, A, L))
        for a, b in product(range(A), range(A)):
            phi_ab_split = np.array([
                [d[b, 0], d[b, 1]],
                [d[b, 0], d[b, 1]],
            ])
            phi_ab_split = phi_ab_split / phi_ab_split.sum(1)
            phi_ab_split = 1 + 0 * phi_ab_split / phi_ab_split.sum(1)
            phi_all_extended[a, :, b, :] = self.phi_all[a, b] * phi_ab_split
            phi_school_extended[a, :, b, :] = self.phi_school[a, b] * phi_ab_split
            phi_work_extended[a, :, b, :] = self.phi_work[a, b] * phi_ab_split
        
        # Apply school closure and social distance
        if day_type == 1:  # Weekday
            phi_age_risk = (1 - social_distance) * (phi_all_extended - school * phi_school_extended)
            if cocooning > 0:
                # Assumes 95% reduction on last age group and high risk
                # High risk cocooning
                phi_age_risk_copy = phi_all_extended - school * phi_school_extended
                phi_age_risk[:, 1, :, :] = (1 - cocooning) * phi_age_risk_copy[:, 1, :, :]
                # last age group cocooning
                phi_age_risk[-1, :, :, :] = (1 - cocooning) * phi_age_risk_copy[-1, :, :, :]
            assert (phi_age_risk >= 0).all()
            return phi_age_risk
        elif day_type == 2 or day_type == 3:  # is a weekend or holiday
            phi_age_risk = (1 - social_distance) * (phi_all_extended - phi_school_extended - phi_work_extended)
            if cocooning > 0:
                # Assumes 95% reduction on last age group and high risk
                # High risk cocooning
                phi_age_risk_copy = (phi_all_extended - phi_school_extended - phi_work_extended)
                phi_age_risk[:, 1, :, :] = (1 - cocooning) * phi_age_risk_copy[:, 1, :, :]
                # last age group cocooning
                phi_age_risk[-1, :, :, :] = (1 - cocooning) * phi_age_risk_copy[-1, :, :, :]
            assert (phi_age_risk >= 0).all()
            return phi_age_risk
        else:
            phi_age_risk = (1 - social_distance) * (phi_all_extended - phi_school_extended)
            if cocooning > 0:
                # Assumes 95% reduction on last age group and high risk
                # High risk cocooning
                phi_age_risk_copy = (phi_all_extended - phi_school_extended)
                phi_age_risk[:, 1, :, :] = (1 - cocooning) * phi_age_risk_copy[:, 1, :, :]
                # last age group cocooning
                phi_age_risk[-1, :, :, :] = (1 - cocooning) * phi_age_risk_copy[-1, :, :, :]
            assert (phi_age_risk >= 0).all()
            return phi_age_risk
        
    def update_hosp_duration(self):
        self.gamma_ICU = self.gamma_ICU0*(1 + self.alpha1)
        self.mu_ICU = self.mu_ICU0*(1 + self.alpha3)
        self.gamma_IH = self.gamma_IH0*(1 - self.alpha2)
    
    def update_icu_params(self, rdrate):
        # update the ICU admission parameter HICUR and update nu
        self.HICUR = self.HICUR * rdrate
        self.nu = self.gamma_IH * self.HICUR / (self.mu + (self.gamma_IH- self.mu) * self.HICUR)
        self.rIH = 1 - (1 - self.rIH)*rdrate
        
    def update_icu_all(self, t, otherInfo):
        if 'rIH' in otherInfo.keys():
            if t in otherInfo['rIH'].keys():
                self.rIH = otherInfo['rIH'][t]
            else:
                self.rIH = self.rIH0
        if 'HICUR' in otherInfo.keys():
            if t in otherInfo['HICUR'].keys():
                self.HICUR = otherInfo['HICUR'][t]
            else:
                self.HICUR = self.HICUR0
        if 'mu' in otherInfo.keys():
            if t in otherInfo['mu'].keys():
                self.mu = self.mu0.copy()/otherInfo['mu'][t]
            else:
                self.mu = self.mu0.copy()
        self.nu = self.gamma_IH * self.HICUR / (self.mu + (self.gamma_IH- self.mu) * self.HICUR)
        

class ParamDistribution():
    '''
        A class to encapsulate epi paramters that are random
        Attrs:
            is_inverse (bool): if True, the parameter is used in the model as 1 / x.
            param_name (str): Name of the parameter, used in EpiParams as attribute name.
            distribution_name (str): Name of the distribution, matching functions in np.random.
            det_val (float): Value of the parameter for deterministic simulations.
            params (list): paramters if the distribution
    '''
    def __init__(self, inv_opt, param_name, distribution_name, det_val, params):
        if inv_opt == "rnd_inverse":
            self.is_inverse = True
        elif inv_opt == "rnd":
            self.is_inverse = False
        self.param_name = param_name
        self.distribution_name = distribution_name
        self.det_val = det_val
        self.params = params
    
    def sample(self, rnd_stream, dim=1):
        '''
            Sample random variable with given distribution name, parameters and dimension.
            Args:
                rnd_stream (np.RandomState): a random stream. If None, det_val is returned.
                dim (int or tuple): dimmention of the parameter (default is 1).
        '''
        if rnd_stream is not None:
            dist_func = getattr(rnd_stream, self.distribution_name)
            args = self.params
            if self.is_inverse:
                return np.squeeze(1 / dist_func(*args, dim))
            else:
                return np.squeeze(dist_func(*args, dim))
        else:
            if self.is_inverse:
                return 1 / self.det_val
            else:
                return self.det_val
