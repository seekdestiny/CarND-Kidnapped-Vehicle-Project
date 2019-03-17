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
  if (is_initialized) {
    return;
  }

  num_particles = 1024;  // TODO: Set the number of particles
  
  particles.resize(num_particles);
  weights.resize(num_particles);

  // sensor noise
  std::default_random_engine gen;
  std::normal_distribution<double> norm_x (x, std[0]);
  std::normal_distribution<double> norm_y (y, std[1]);
  std::normal_distribution<double> norm_theta(theta, std[2]);

  // initialize particle
  for (auto i = 0; i < num_particles; i++) {
    weights[i] = 1.0;
    particles[i].x = norm_x(gen);
    particles[i].y = norm_y(gen);
    particles[i].theta = norm_theta(gen);
    particles[i].weight = weights[i];
    particles[i].id = i;
  }
  
  // finish initialization
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

  // for each particle predict its position and Gaussian noise
  for (Particle &particle: particles) {
    // Predict
    if (fabs(yaw_rate) < 0.00001) {
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    } else {
      double theta_new = particle.theta + yaw_rate * delta_t;
      double velocity_div_yaw_rate = velocity / yaw_rate;
      particle.x += velocity_div_yaw_rate * ( std::sin(theta_new) - std::sin(particle.theta));
      particle.y += velocity_div_yaw_rate * ( std::cos(particle.theta) - std::cos(theta_new));
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

      if (cur_dist <= min_dist) {
        min_dist = cur_dist;
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
  for (unsigned int j = 0; j < particles.size(); j++) {
    Particle const &particle = particles[j];
    //
    // (1) transform observations to the map coordinates
    //

    vector<LandmarkObs> transformed_observations(observations.size());
    for (unsigned int i = 0; i < observations.size(); i++) {
      double cos_theta = cos(particle.theta);
      double sin_theta = sin(particle.theta);

      LandmarkObs observation = observations[i];
      transformed_observations[i].x = particle.x + cos_theta * observation.x - sin_theta * observation.y;
      transformed_observations[i].y = particle.y + sin_theta * observation.x + cos_theta * observation.y;
      transformed_observations[i].id = -1;  // we do not know with which landmark to associate this observation yet
    }


    //
    // (2) associate each transformed observation with a landmark identifier
    //

    // make an array with landmarks that are within the sensor range
    vector<LandmarkObs> landmarks;
    for (auto const &landmark : map_landmarks.landmark_list) {
      if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= sensor_range) {
        LandmarkObs lm_obs = {
            .id = landmark.id_i,
            .x = static_cast<double>(landmark.x_f),
            .y = static_cast<double>(landmark.y_f),
        };
        landmarks.push_back(lm_obs);
      }
    }

    // associate transformed observations with landmarks
    dataAssociation(landmarks, transformed_observations);


    //
    // (3) update particle's weight
    //

    // (3.1) determine measurement probabilities
    vector<double> observation_probabilities(transformed_observations.size());
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    particles[j].weight = 1.0;  // set to 1 for multiplication in the end of the loop
    for (unsigned int i = 0; i < observations.size(); i++) {
      LandmarkObs tobs = transformed_observations[i];
      LandmarkObs nearest_landmark = {
          .id = -1,  // not important here
          .x = static_cast<double>(map_landmarks.landmark_list[tobs.id - 1].x_f), // landmark indices start at 1
          .y = static_cast<double>(map_landmarks.landmark_list[tobs.id - 1].y_f),
      };

      // helper variables
      double x_diff_2 = pow(tobs.x - nearest_landmark.x, 2.0);
      double y_diff_2 = pow(tobs.y - nearest_landmark.y, 2.0);
      double std_x_2 = pow(std_landmark[0], 2.0);
      double std_y_2 = pow(std_landmark[1], 2.0);

      // formula of multivariate Gaussian probability
      observation_probabilities[i] = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1])) *
                                     exp(-(x_diff_2 / (2 * std_x_2) + y_diff_2 / (2 * std_y_2)));

      // (3.2) combine probabilities (particle's final weight)
      particles[j].weight *= observation_probabilities[i];
      associations.push_back(tobs.id);
      sense_x.push_back(tobs.x);
      sense_y.push_back(tobs.y);
    }

    // set calculated particle weight in the weights array
    weights[j] = particles[j].weight;
    SetAssociations(particles[j], associations, sense_x, sense_y);
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

  // Get weights and max weight.
  vector<double> weights;
  double maxWeight = std::numeric_limits<double>::min();
  for(int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
    if ( particles[i].weight > maxWeight ) {
      maxWeight = particles[i].weight;
    }
  }

  // Creating distributions.
  std::uniform_real_distribution<double> distDouble(0.0, maxWeight);
  std::uniform_int_distribution<int> distInt(0, num_particles - 1);

  // Generating index.
  int index = distInt(gen);

  double beta = 0.0;

  // the wheel
  vector<Particle> resampledParticles;
  for(int i = 0; i < num_particles; i++) {
    beta += distDouble(gen) * 2.0;
    while( beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }

  particles = resampledParticles;
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
