#@title contextual decision masking task

import numpy as np
import neurogym as ngym
from neurogym.utils import spaces


class SimpleDM(ngym.core.TrialEnv):
    """Simple DM task
    """
    metadata = {
        'paper_link': None,
        'paper_name': None,
        'tags': [],
    }

    def __init__(self, dt=20, rewards=None, timing=None, stim_strengths=None, noise_std=None):
        super().__init__(dt=dt)

        if stim_strengths is None:
            self.stim_strengths = [-0.4, -0.2, -0.1, 0.1, 0.2, 0.4]
        else:
            self.stim_strengths = stim_strengths
        if noise_std is None:
            self.noise_std = 0.1
        else:
            self.noise_std = noise_std

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 200,
            'delay1': 100,
            'stim': 1000,
            'delay2': 100,
            'decision': 400}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # action and observation space
        name = {
            'fixation': 0,
            'context': 1,
            'stimulus_mod1': 2,
        }
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3,),
            dtype=np.float32, name=name)
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape=(1,),
            dtype=np.float32, name={'choice': 0})

    def _new_trial(self, **kwargs):
        trial = {}
        stim1 = self.rng.choice(self.stim_strengths)
        target = stim1 > 0
        target = 1 if target else -1
        trial['stim1'] = stim1

        # Periods
        periods = ['fixation', 'delay1', 'stim', 'delay2', 'decision']
        self.add_period(periods)

        self.add_ob(1, period=['fixation', 'delay1', 'stim', 'delay2'], where='fixation')
        self.add_ob(1, period=periods, where='context')
        self.add_ob(stim1, period='stim', where='stimulus_mod1')

        self.set_groundtruth(target, period='decision')

        return trial

    def _step(self, action):
        new_trial = False
        gt = self.gt_now
        ob = self.ob_now
        # rewards
        reward = 0
        if not self.in_period('decision'):
            if np.abs(action - 0) > 1e-1:
                new_trial = self.abort
                reward = self.rewards['abort']
        else:
            if np.abs(action - gt) < 1e-1:
                reward = self.rewards['correct']
                new_trial = True
            else:
                reward = self.rewards['fail']

        return ob, reward, False, False, {'new_trial': new_trial, 'gt': gt}


class MultiDM(ngym.core.TrialEnv):
    """Multi DM task
    """

    def __init__(
        self,
        dt=50,
        rewards=None,
        timing=None,
        stim_strengths=None,
        stim_durs=None,
        noise_std=None,
        signal="mean",
        target="1",
        thresholds=None,
    ):
        super().__init__(dt=dt)

        if stim_strengths is None:
            self.stim_strengths = [-0.8, -0.4, -0.2, 0.2, 0.4, 0.8]
        else:
            self.stim_strengths = stim_strengths
        if noise_std is None:
            self.noise_std = 0.1
        else:
            self.noise_std = noise_std
        if stim_durs is None:
            self.stim_durs = [350, 500, 700, 850]
        else:
            self.stim_durs = stim_durs
            assert np.max(self.stim_durs) <= 1000
            assert np.min(self.stim_durs) >= 100
        if thresholds is None:
            self.thresholds = [-0.5, 0, 0.5]
        else:
            self.thresholds = thresholds

        # task config
        assert signal in ["mean", "time", "integral"]
        assert target in ["1", "2", "sum", "diff"]
        self.signal = signal
        self.target = target

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 200,
            'delay1': 100,
            'stim': 1000,
            'delay2': 100,
            'decision': 600}
        if timing:
            raise ValueError("MultiDM does not support timing")

        self.abort = False

        # action and observation space
        name = {
            'fixation': 0,
            'context': 1,
            'threshold': 2,
            'stimulus_mod1': 3,
            'stimulus_mod2': 4,
        }
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(5,),
            dtype=np.float32, name=name)
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape=(1,),
            dtype=np.float32, name={'choice': 0})

    def _new_trial(self, **kwargs):
        trial = {}
        stim1, stim2 = self.rng.choice(np.unique(np.abs(self.stim_strengths)), size=2, replace=False)
        stim1 = stim1 * self.rng.choice([-1, 1])
        stim2 = stim2 * self.rng.choice([-1, 1])
        dur1, dur2 = self.rng.choice(self.stim_durs, size=2, replace=False)
        threshold = self.rng.choice(self.thresholds)
        trial['stim1'] = stim1
        trial['stim2'] = stim2
        trial['dur1'] = dur1
        trial['dur2'] = dur2
        binarize = lambda val: 1 if val else -1
        if self.signal == "mean":
            eff_threshold = threshold * 0.6  # [-0.3, 0, 0.3]
            if self.target == "1":
                target = binarize(stim1 > eff_threshold)
            elif self.target == "2":
                target = binarize(stim2 > eff_threshold)
            elif self.target == "sum":
                target = binarize((stim1 + stim2) > eff_threshold * 2)
            elif self.target == "diff":
                target = binarize((stim1 - stim2) > eff_threshold * 2)
        elif self.signal == "time":
            eff_threshold = threshold * 250 + 600  # [475, 600, 725]
            if self.target == "1":
                target = binarize(dur1 > eff_threshold)
            elif self.target == "2":
                target = binarize(dur2 > eff_threshold)
            elif self.target == "sum":
                target = binarize((dur1 + dur2) > eff_threshold * 2)
            elif self.target == "diff":
                target = binarize((dur1 - dur2) > (eff_threshold - 600) * 2)
        elif self.signal == "integral":
            threshold = threshold * 400  # [-200, 0, 200]
            if self.target == "1":
                target = binarize(stim1 * dur1 > threshold)
            elif self.target == "2":
                target = binarize(stim2 * dur2 > threshold)
            elif self.target == "sum":
                target = binarize((stim1 * dur1 + stim2 * dur2) > threshold * 2)
            elif self.target == "diff":
                target = binarize((stim1 * dur1 - stim2 * dur2) > threshold * 2)
        else:
            raise ValueError("Unknown signal")
        trial['threshold'] = threshold
        trial['target'] = target

        # Periods
        periods = ['fixation', 'delay1', 'stim', 'delay2', 'decision']
        self.add_period(periods)

        self.add_ob(1, period=['fixation', 'delay1', 'stim', 'delay2'], where='fixation')
        self.add_ob(1, period=periods, where='context')
        # self.add_ob(threshold, period=['delay2', 'decision'], where='threshold')
        self.add_ob(threshold, period=periods, where='threshold')

        total_stim_dur = self.view_ob(period='stim').shape[0]
        stim1_arr = np.zeros(total_stim_dur)
        stim2_arr = np.zeros(total_stim_dur)
        dur1_samp = int(total_stim_dur * dur1 / 1000)
        dur2_samp = int(total_stim_dur * dur2 / 1000)
        dur1_offset = 0  # self.rng.randint(0, total_stim_dur - dur1_samp)
        dur2_offset = 0  # self.rng.randint(0, total_stim_dur - dur2_samp)
        stim1_arr[dur1_offset:dur1_offset+dur1_samp] = stim1
        stim2_arr[dur2_offset:dur2_offset+dur2_samp] = stim2
        self.add_ob(stim1_arr, period='stim', where='stimulus_mod1')
        self.add_ob(stim2_arr, period='stim', where='stimulus_mod2')

        self.set_groundtruth(target, period='decision')

        return trial

    def _step(self, action):
        new_trial = False
        gt = self.gt_now
        ob = self.ob_now
        # rewards
        reward = 0
        if not self.in_period('decision'):
            if np.abs(action - 0) > 1e-1:
                new_trial = self.abort
                reward = self.rewards['abort']
        else:
            if np.abs(action - gt) < 1e-1:
                reward = self.rewards['correct']
                new_trial = True
            else:
                reward = self.rewards['fail']

        return ob, reward, False, False, {'new_trial': new_trial, 'gt': gt}


class MultiDM2(ngym.core.TrialEnv):
    """Multi DM task
    """

    def __init__(
        self,
        dt=50,
        rewards=None,
        timing=None,
        stim_means=None,
        stim_freqs=None,
        stim_amps=None,
        signal="mean",
        target="1",
    ):
        super().__init__(dt=dt)

        if stim_means is None:
            self.stim_means = [-0.8, -0.4, -0.2, 0.2, 0.4, 0.8]
        else:
            self.stim_means = stim_means    
        if stim_freqs is None:
            self.stim_freqs = [1,2,3,4,5,6]
        else:
            self.stim_freqs = stim_freqs
            assert np.max(self.stim_freqs) <= 1000 / dt
            assert np.min(self.stim_freqs) >= 1
            assert len(self.stim_freqs) == 6, "Invalid number of stimulus frequencies (for now)"
        if stim_amps is None:
            self.stim_amps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        else:
            self.stim_amps = stim_amps
            assert np.max(self.stim_amps) <= 1.0
            assert np.min(self.stim_amps) >= 0.0
            assert len(self.stim_amps) == 6, "Invalid number of stimulus amplitudes (for now)"

        # task config
        assert signal in ["mean", "freq", "amp"]
        assert target in ["1", "2", "sum", "diff"]
        self.signal = signal
        self.target = target

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 200,
            'delay1': 100,
            'stim': 1000,
            'delay2': 100,
            'decision': 600}
        if timing:
            raise ValueError("MultiDM does not support timing")

        self.abort = False

        # action and observation space
        name = {
            'fixation': 0,
            'context': 1,
            'stimulus_mod1': 2,
            'stimulus_mod2': 3,
        }
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(4,),
            dtype=np.float32, name=name)
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape=(1,),
            dtype=np.float32, name={'choice': 0})

    def _new_trial(self, **kwargs):
        trial = {}
        stim1_mean, stim2_mean = self.rng.choice(np.unique(np.abs(self.stim_means)), size=2, replace=False)
        stim1_mean = stim1_mean * self.rng.choice([-1, 1])
        stim2_mean = stim2_mean * self.rng.choice([-1, 1])
        stim1_freq, stim2_freq = self.rng.choice(3, size=2, replace=False)
        stim1_freq = (stim1_freq + 0.5) * self.rng.choice([-1, 1]) + 2.5
        stim2_freq = (stim2_freq + 0.5) * self.rng.choice([-1, 1]) + 2.5
        stim1_freq = self.stim_freqs[int(stim1_freq)]
        stim2_freq = self.stim_freqs[int(stim2_freq)]
        stim1_amp, stim2_amp = self.rng.choice(3, size=2, replace=False)
        stim1_amp = (stim1_amp + 0.5) * self.rng.choice([-1, 1]) + 2.5
        stim2_amp = (stim2_amp + 0.5) * self.rng.choice([-1, 1]) + 2.5
        stim1_amp = self.stim_amps[int(stim1_amp)]
        stim2_amp = self.stim_amps[int(stim2_amp)]
        trial['stim1_mean'] = stim1_mean
        trial['stim2_mean'] = stim2_mean
        trial['stim1_freq'] = stim1_freq
        trial['stim2_freq'] = stim2_freq
        trial['stim1_amp'] = stim1_amp
        trial['stim2_amp'] = stim2_amp
        binarize = lambda val: 1 if val else -1
        if self.signal == "mean":
            if self.target == "1":
                target = binarize(stim1_mean > 0)
            elif self.target == "2":
                target = binarize(stim2_mean > 0)
            elif self.target == "sum":
                target = binarize((stim1_mean + stim2_mean) > 0)
            elif self.target == "diff":
                target = binarize((stim1_mean - stim2_mean) > 0)
        elif self.signal == "freq":
            if self.target == "1":
                target = binarize(stim1_freq > np.median(self.stim_freqs))
            elif self.target == "2":
                target = binarize(stim2_freq > np.median(self.stim_freqs))
            elif self.target == "sum":
                target = binarize((stim1_freq + stim2_freq) > (np.median(self.stim_freqs) * 2))
            elif self.target == "diff":
                target = binarize((stim1_freq - stim2_freq) > 0)
        elif self.signal == "amp":
            if self.target == "1":
                target = binarize(stim1_amp > np.median(self.stim_amps))
            elif self.target == "2":
                target = binarize(stim2_amp > np.median(self.stim_amps))
            elif self.target == "sum":
                target = binarize((stim1_amp + stim2_amp) > (np.median(self.stim_amps) * 2))
            elif self.target == "diff":
                target = binarize((stim1_amp - stim2_amp) > 0)
        else:
            raise ValueError("Unknown signal")
        trial['target'] = target

        # Periods
        periods = ['fixation', 'delay1', 'stim', 'delay2', 'decision']
        self.add_period(periods)

        self.add_ob(1, period=['fixation', 'delay1', 'stim', 'delay2'], where='fixation')
        self.add_ob(1, period=periods, where='context')
        # self.add_ob(threshold, period=['delay2', 'decision'], where='threshold')

        total_stim_dur = self.view_ob(period='stim').shape[0]
        true_stim_dur = total_stim_dur * self.dt / 1000  # dur in s
        timestamps = np.arange(0, true_stim_dur, self.dt / 1000)[:total_stim_dur]
        stim1_arr = np.zeros(total_stim_dur)
        stim2_arr = np.zeros(total_stim_dur)
        stim1_arr = stim1_arr + stim1_mean
        stim2_arr = stim2_arr + stim2_mean
        stim1_arr = stim1_arr + np.sin(timestamps * stim1_freq * 2 * np.pi) * stim1_amp
        stim2_arr = stim2_arr + np.sin(timestamps * stim2_freq * 2 * np.pi) * stim2_amp
        self.add_ob(stim1_arr, period='stim', where='stimulus_mod1')
        self.add_ob(stim2_arr, period='stim', where='stimulus_mod2')

        self.set_groundtruth(target, period='decision')

        return trial

    def _step(self, action):
        new_trial = False
        gt = self.gt_now
        ob = self.ob_now
        # rewards
        reward = 0
        if not self.in_period('decision'):
            if np.abs(action - 0) > 1e-1:
                new_trial = self.abort
                reward = self.rewards['abort']
        else:
            if np.abs(action - gt) < 1e-1:
                reward = self.rewards['correct']
                new_trial = True
            else:
                reward = self.rewards['fail']

        return ob, reward, False, False, {'new_trial': new_trial, 'gt': gt}


class ContextDM(ngym.core.TrialEnv):
    """Context DM task
    """
    metadata = {
        'paper_link': None,
        'paper_name': None,
        'tags': [],
    }

    def __init__(self, dt=20, rewards=None, timing=None, stim_strengths=None, noise_std=None):
        super().__init__(dt=dt)

        if stim_strengths is None:
            self.stim_strengths = [-0.4, -0.2, -0.1, 0.1, 0.2, 0.4]
        else:
            self.stim_strengths = stim_strengths
        if noise_std is None:
            self.noise_std = 0.1
        else:
            self.noise_std = noise_std

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': lambda: self.rng.uniform(200, 600),
            'delay1': lambda: self.rng.uniform(100, 300),
            'stim': lambda: self.rng.uniform(200, 1600),
            'delay2': lambda: self.rng.uniform(100, 300),
            'decision': lambda: self.rng.uniform(300, 700),}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # action and observation space
        name = {
            'fixation': 0,
            'context': range(1, 3),
            'stimulus_mod1': 3,
            'stimulus_mod2': 4,
        }
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(5,),
            dtype=np.float32, name=name)
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape=(1,),
            dtype=np.float32, name={'choice': 0})

    def _new_trial(self, **kwargs):
        trial = {}
        stim1 = self.rng.choice(self.stim_strengths)
        stim2 = self.rng.choice(self.stim_strengths)
        i_context = self.rng.choice([0, 1])
        context = np.array([0, 1]) if i_context == 1 else np.array([1, 0])
        target = stim1 > 0 if i_context == 0 else stim2 > 0
        target = 1 if target else -1
        trial['stim1'] = stim1
        trial['stim2'] = stim2
        trial['context'] = context

        # Periods
        periods = ['fixation', 'delay1', 'stim', 'delay2', 'decision']
        self.add_period(periods)

        self.add_ob(1, period=['fixation', 'delay1', 'stim', 'delay2'], where='fixation')
        self.add_ob(context, period=['delay1', 'stim', 'delay2'], where='context')
        self.add_ob(stim1, period='stim', where='stimulus_mod1')
        self.add_ob(stim2, period='stim', where='stimulus_mod2')

        self.set_groundtruth(target, period='decision')

        return trial

    def _step(self, action):
        new_trial = False
        gt = self.gt_now
        ob = self.ob_now
        # rewards
        reward = 0
        if not self.in_period('decision'):
            if np.abs(action - 0) > 1e-1:
                new_trial = self.abort
                reward = self.rewards['abort']
        else:
            if np.abs(action - gt) < 1e-1:
                reward = self.rewards['correct']
                new_trial = True
            else:
                reward = self.rewards['fail']

        return ob, reward, False, False, {'new_trial': new_trial, 'gt': gt}
