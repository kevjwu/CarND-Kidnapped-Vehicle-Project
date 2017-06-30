/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"

using namespace std;

default_random_engine _GEN;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 15;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    for (int i=0; i < num_particles; i++){
        double x = dist_x(_GEN);
        double y = dist_y(_GEN);
        double theta = dist_theta(_GEN);
        
        Particle p = {i, x, y, theta};
        particles.push_back(p);
    }
    is_initialized = true;
    return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    
    double a = yaw_rate * delta_t;
    double b = velocity / yaw_rate;
    
    if (fabs(yaw_rate) < 0.001){
        for (vector<Particle>::iterator p = particles.begin() ; p < particles.end(); p++) {
            {
                p->x += velocity * delta_t * cos(p->theta) + dist_x(_GEN);
                p->y += velocity * delta_t * sin(p->theta) + dist_y(_GEN);
                p->theta += dist_theta(_GEN);
            }
        }
    }
    else {
        for (vector<Particle>::iterator p = particles.begin() ; p < particles.end(); p++) {
            {
                
                p->x += b * (sin(p->theta + a) - sin(p->theta)) + dist_x(_GEN);
                p->y += b * (cos(p->theta) - cos(p->theta + a)) + dist_y(_GEN);
                p->theta += a + dist_theta(_GEN);
            }
        }
    }
    return;

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	
    for (int i=0; i<observations.size(); i++){
        double min_dist = __DBL_MAX__;
        for (int j=0; j<predicted.size(); j++){
            double dist = sqrt(pow(observations[i].x - predicted[j].x, 2) + pow(observations[i].y - predicted[j].y, 2));
            if (dist < min_dist){
                min_dist = dist;
                observations[i].id = j;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

    double a= 1 / ( 2 * M_PI * std_landmark[0] * std_landmark[1]);
    double b = 2 * pow(std_landmark[0], 2);
    double c = 2 * pow(std_landmark[1], 2);
    
    
    vector<LandmarkObs> transformed_obs (observations.size());
    double probability;
    
    for (vector<Particle>::iterator p = particles.begin() ; p < particles.end(); p++) {
        
        vector<LandmarkObs> range_landmarks;
        
        for (int i = 0; i < map_landmarks.landmark_list.size(); i++){
            
            double dist = sqrt(pow(map_landmarks.landmark_list[i].x_f - p->x, 2) +
                               pow(map_landmarks.landmark_list[i].y_f - p->y, 2));
            if (dist < sensor_range){
                LandmarkObs lm;
                lm.id = map_landmarks.landmark_list[i].id_i;
                lm.x = map_landmarks.landmark_list[i].x_f;
                lm.y = map_landmarks.landmark_list[i].y_f;
                range_landmarks.push_back(lm);
            }
        }
        
        for (int i=0; i< observations.size(); i++){
            LandmarkObs pred;
            pred.id = i;
            pred.x = (observations[i].x * cos(p->theta) - observations[i].y * sin(p->theta) + p->x);
            pred.y = (observations[i].x * sin(p->theta) + observations[i].y * cos(p->theta) + p->y);
            transformed_obs[i] = pred;
        }
        
        vector<LandmarkObs> associated_obs = transformed_obs;
        
        
        dataAssociation(range_landmarks, associated_obs);
        
        probability = 1;
        for (int i=0; i < associated_obs.size(); i++){
            probability *= a * exp(-1 * (pow(range_landmarks[associated_obs[i].id].x - associated_obs[i].x, 2)/ b + pow(range_landmarks[associated_obs[i].id].y - associated_obs[i].y, 2)/ c ));
        }
        p->weight = probability;
        
    }
}

void ParticleFilter::resample() {
	vector<double> weights (particles.size());
    for (int i=0; i<particles.size(); i++) {
        weights[i] = particles[i].weight;
    }
    discrete_distribution<> dist(weights.begin(), weights.end());
    vector<Particle> resampled_particles = particles;
    for (int i=0; i < particles.size() ; i++) {
        Particle p = particles[dist(_GEN)];
        p.id = i;
        resampled_particles[i] = p;
    }
    particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
