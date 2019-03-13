/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  particles.resize(num_particles);
  weights.resize(num_particles);

  // sensor noise
  std::default_random_engine gen;
  std::normal_distribution<double> norm_x (x, std[0]);
  std::normal_distribution<double> norm_y (y, std[1]);
  std::normal_distribution<double> norm_theta(theta, std[2]);

  int id = 0;
  for (Particle &particle : particles) {
      particle.id = id;
      particle.x = norm_x(gen);
      particle.y = norm_y(gen);
      particle.theta = norm_theta(gen);
      particle.weight = 1.0;
      ++id;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  // sensor noise
  std::default_random_engine gen;
  std::normal_distribution<double> norm_x (0, std_pos[0]);
  std::normal_distribution<double> norm_y (0, std_pos[1]);
  std::normal_distribution<double> norm_theta(0, std_pos[2]);

  for (Particle &particle: particles) {
    // Predict
    double theta_new = particle.theta + yaw_rate * delta_t;


    if (fabs(yaw_rate) < 0.00001) {
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    } else {
      particle.x += (velocity / yaw_rate) * ( std::sin(theta_new) - std::sin(particle.theta));
      particle.y += (velocity / yaw_rate) * ( std::sin(particle.theta) - std::sin(theta_new));
      particle.theta = theta_new;
    }
  
    // add noise
    particle.x += norm_x(gen);
    particle.y += norm_y(gen);
    particle.theta += norm_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
 
   for (unsigned int i = 0; i < observations.size(); ++i) {
    const LandmarkObs& ob = observations[i];

    double min_dist = std::numeric_limits<double>::max();
    
    int map_id = -1;

    for (unsigned int j = 0; j < predicted.size(); ++j) {
      const LandmarkObs& pred = predicted[j];

      double cur_dist = dist(ob.x, ob.y, pred.x, pred.y);

      if (cur_dist < min_dist) {
        cur_dist = min_dist;
        map_id = pred.id;
      }
    }

    observations[i].id = map_id;
   } 
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

    // for each particle
    for (int i = 0; i < num_particles; i++) {
      double p_x = particles[i].x;
      double p_y = particles[i].y;
      double p_theta = particles[i].theta;

      // vector to hold map landmark locations predicted to be within sensor range
      vector<LandmarkObs> predicted;

      for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
        float lm_x = map_landmarks.landmark_list[j].x_f;
        float lm_y = map_landmarks.landmark_list[j].y_f;
        int lm_id = map_landmarks.landmark_list[j].id_i;

        if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) {
          predicted.push_back(LandmarkObs{lm_id, lm_x, lm_y});
        }
      }

      vector<LandmarkObs> transformed_os;
      for (unsigned int j = 0; j < observations.size(); j++) {
        double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
        double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
        transformed_os.push_back(LandmarkObs{observations[j].id, t_x, t_y});
      }

      dataAssociation(predicted, transformed_os);

      particles[i].weight = 1.0;

      for (unsigned int j = 0; j < transformed_os.size(); j++) {
        double o_x = transformed_os[j].x;
        double o_y = transformed_os[j].y;
        double pr_x, pr_y;

        int associated_prediction = transformed_os[j].id;

        for (unsigned int k = 0; k < predicted.size(); k++) {
          if (predicted[k].id == associated_prediction) {
              pr_x = predicted[k].x;
              pr_y = predicted[k].y;
          } 
        }

        double s_x = std_landmark[0];
        double s_y = std_landmark[1];
        double obs_w = (1 / (2 * M_PI * s_x * s_y)) * exp( -(pow(pr_x - o_x, 2) / (2 * pow(s_x, 2)) + 
                                                             pow(pr_y - o_y, 2) / (2 * pow(s_y, 2))));

        particles[i].weight *= obs_w;
      }
    }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;

  vector<Particle> new_particles;

  // get all of the current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // generate random starting index for resampling wheel
  std::uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  // uniform random distribution [0.0, max_weight)
  std::uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  // spin the resample wheel!
  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
