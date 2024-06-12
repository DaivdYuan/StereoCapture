import numpy as np
import cv2

class ParticleFilter3D:
    def __init__(self, num_particles, val_range, diffusion_rate=0.1, random_resample_rate=0.1, base_value=0.01, retry = 5):
        assert len(val_range) == 3
        
        self.num_particles = num_particles
        self.val_range = val_range
        self.diffusion_rate = diffusion_rate
        self.random_resample_rate = random_resample_rate
        self.base_value = base_value
        self.retry = retry
        
        self.particles = (np.random.rand(num_particles, 3) - 0.5) * np.array(val_range) * 2
        self.weights = np.ones(num_particles) / num_particles
        
        self.camera_inited = False
        
    def init_camera(self, mtx, dist, rl, tl, rm, tm, rr, tr):
        self.mtx = mtx
        self.dist = dist
        self.r = {
            'l': rl,
            'm': rm,
            'r': rr
        }
        self.t = {
            'l': tl,
            'm': tm,
            'r': tr
        }
        
        self.camera_inited = True
        
    def diffuse(self):
        # diffuse using gaussian noise
        self.particles += (np.random.randn(self.num_particles, 3) * self.diffusion_rate)
        
    def resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

        if self.random_resample_rate > 0:        
            resample_num = int(self.num_particles * self.random_resample_rate)
            self.particles[:resample_num] = (np.random.rand(resample_num, 3) - 0.5) * np.array(self.val_range) * 2
            
        self.diffuse()
            
    def update_gaussian_3d(self, mean, cov):
        assert mean.shape == (3,)
        assert cov.shape == (3,)
        self.weights *= (np.exp(-0.5 * np.sum((self.particles - mean) ** 2 / cov, axis=1)) + self.base_value)
        self.weights /= np.sum(self.weights)
        
    def update_gaussian_2d(self, points, mean, cov):
        assert mean.shape == (2,)
        assert cov.shape == (2,)
        
        self.weights *= (np.exp(-0.5 * np.sum((points - mean) ** 2 / cov, axis=1)) + self.base_value)
        self.weights /= np.sum(self.weights)
        
    def project_and_update_l2d(self, means):
        loss = np.zeros(self.num_particles)
        if 'l' in means:
            self.projected_points_l, _ = cv2.projectPoints(self.particles, self.r['l'], self.t['l'], self.mtx, self.dist)
            self.projected_points_l = self.projected_points_l.reshape(-1, 2)
            loss += np.sum((self.projected_points_l - means['l']) ** 2, axis=1)
        if 'm' in means:
            self.projected_points_m, _ = cv2.projectPoints(self.particles, self.r['m'], self.t['m'], self.mtx, self.dist)
            self.projected_points_m = self.projected_points_m.reshape(-1, 2)
            loss += np.sum((self.projected_points_m - means['m']) ** 2, axis=1)
        if 'r' in means:
            self.projected_points_r, _ = cv2.projectPoints(self.particles, self.r['r'], self.t['r'], self.mtx, self.dist)
            self.projected_points_r = self.projected_points_r.reshape(-1, 2)
            loss += np.sum((self.projected_points_r - means['r']) ** 2, axis=1)
            
        self.weights *= (5 * np.exp(-0.5 * loss) + self.base_value)
        self.weights /= np.sum(self.weights)
    
    def get_estimate(self, num_top_particles = 10):
        indices = np.argsort(self.weights)[::-1][:num_top_particles]
        return np.average(self.particles[indices], axis=0, weights=self.weights[indices])
    
    def get_top_particles(self, num_top_particles = 10):
        indices = np.argsort(self.weights)[::-1]
        return self.particles[indices[:num_top_particles]]
    
    def step_filter_3d(self, means, covs):
        self.resample()
        for mean, cov in zip(means, covs):
            self.update_gaussian_3d(mean, cov)
        return self.get_estimate()
    
    def step_filter_2d(self, means, covs):
        if not self.camera_inited:
            print("Camera not initialized")
            return None
        
        cnt = 0
        while cnt < self.retry:
            self.resample()
            
            self.project_and_update_l2d(means)
            if 'l' in means:
                self.update_gaussian_2d(self.projected_points_l, means['l'], covs['l'])
            if 'm' in means:
                self.update_gaussian_2d(self.projected_points_m, means['m'], covs['m'])
            if 'r' in means:
                self.update_gaussian_2d(self.projected_points_r, means['r'], covs['r'])
            
            cnt += 1
            top_points = self.get_top_particles(20)
            variance = np.var(top_points, axis=0)
            
            if np.all(variance < 1):
                break
            
        return top_points, variance
    
if __name__ == '__main__':
    num_particles = 10000
    val_range = [10, 10, 10]
    diffusion_rate = 0.1
    random_resample_rate = 0.1
    
    pf = ParticleFilter3D(num_particles, val_range, diffusion_rate, random_resample_rate)
    print(pf.get_estimate())
    
    means = np.array([[5, 5, 5], [0, 0, 0]])
    covs = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    
    for i in range(100):
        pf.step_filter(means, covs)
        print(pf.get_top_particles(10))
        
        
        