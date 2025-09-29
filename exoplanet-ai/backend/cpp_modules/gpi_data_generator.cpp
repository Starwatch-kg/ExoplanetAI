/*
GPI Data Generator - C++ module for generating high-precision synthetic GPI data
Generates realistic gravitational phase interferometry data for testing and development
*/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <complex>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class GPIDataGenerator {
private:
    std::mt19937 rng;
    std::normal_distribution<double> noise_dist;
    std::uniform_real_distribution<double> uniform_dist;
    
    // Physical constants
    static constexpr double G = 6.67430e-11;  // Gravitational constant
    static constexpr double c = 299792458.0;   // Speed of light
    static constexpr double AU = 1.496e11;     // Astronomical unit
    static constexpr double M_sun = 1.989e30;  // Solar mass
    static constexpr double M_earth = 5.972e24; // Earth mass
    
public:
    GPIDataGenerator(unsigned int seed = 0) : 
        rng(seed ? seed : std::chrono::steady_clock::now().time_since_epoch().count()),
        noise_dist(0.0, 1.0),
        uniform_dist(0.0, 1.0) {}
    
    struct PlanetarySystem {
        double star_mass;           // Solar masses
        double star_radius;         // Solar radii
        double planet_mass;         // Earth masses
        double planet_radius;       // Earth radii
        double orbital_period;      // Days
        double semi_major_axis;     // AU
        double eccentricity;
        double inclination;         // Radians
        double phase_offset;        // Radians
        double distance_to_system;  // Parsecs
    };
    
    struct GPISignature {
        std::vector<double> time_points;
        std::vector<double> phase_shifts;
        std::vector<double> amplitude_variations;
        std::vector<std::complex<double>> gravitational_field;
        std::vector<double> noise_levels;
        double detection_snr;
        double confidence_level;
    };
    
    // Generate realistic planetary system parameters
    PlanetarySystem generateRealisticSystem() {
        PlanetarySystem system;
        
        // Star properties (main sequence stars)
        system.star_mass = 0.5 + uniform_dist(rng) * 1.5;  // 0.5-2.0 solar masses
        system.star_radius = std::pow(system.star_mass, 0.8); // Mass-radius relation
        
        // Planet properties
        system.planet_mass = 0.1 + uniform_dist(rng) * 20.0;  // 0.1-20 Earth masses
        system.planet_radius = std::pow(system.planet_mass / 5.5, 1.0/3.0); // Density-based
        
        // Orbital properties
        system.orbital_period = 1.0 + uniform_dist(rng) * 999.0;  // 1-1000 days
        
        // Kepler's third law: P² ∝ a³/M
        system.semi_major_axis = std::pow(
            (system.orbital_period / 365.25) * (system.orbital_period / 365.25) * system.star_mass,
            1.0/3.0
        );
        
        system.eccentricity = uniform_dist(rng) * 0.3;  // 0-0.3 eccentricity
        system.inclination = std::acos(1.0 - 2.0 * uniform_dist(rng)); // Random inclination
        system.phase_offset = uniform_dist(rng) * 2.0 * M_PI;
        system.distance_to_system = 10.0 + uniform_dist(rng) * 490.0;  // 10-500 parsecs
        
        return system;
    }
    
    // Calculate gravitational phase shift at given time
    double calculatePhaseShift(const PlanetarySystem& system, double time_days) {
        // Convert time to orbital phase
        double orbital_phase = 2.0 * M_PI * time_days / system.orbital_period + system.phase_offset;
        
        // Calculate planet position (simplified circular orbit)
        double x = system.semi_major_axis * AU * std::cos(orbital_phase);
        double y = system.semi_major_axis * AU * std::sin(orbital_phase) * std::sin(system.inclination);
        double z = system.semi_major_axis * AU * std::sin(orbital_phase) * std::cos(system.inclination);
        
        // Distance from observer to planet
        double distance_to_planet = system.distance_to_system * 3.086e16; // Convert parsecs to meters
        double planet_distance = std::sqrt(x*x + y*y + (z + distance_to_planet)*(z + distance_to_planet));
        
        // Gravitational potential difference
        double planet_mass_kg = system.planet_mass * M_earth;
        double gravitational_potential = G * planet_mass_kg / planet_distance;
        
        // Phase shift due to gravitational time dilation
        double phase_shift = 2.0 * gravitational_potential / (c * c);
        
        // Add relativistic corrections
        double velocity = 2.0 * M_PI * system.semi_major_axis * AU / (system.orbital_period * 24 * 3600);
        double doppler_factor = velocity / c;
        phase_shift += doppler_factor * std::cos(orbital_phase);
        
        return phase_shift;
    }
    
    // Generate complete GPI signature
    GPISignature generateGPISignature(const PlanetarySystem& system, int num_points = 10000, double observation_time_days = 365.0) {
        GPISignature signature;
        
        // Generate time points
        signature.time_points.reserve(num_points);
        signature.phase_shifts.reserve(num_points);
        signature.amplitude_variations.reserve(num_points);
        signature.gravitational_field.reserve(num_points);
        signature.noise_levels.reserve(num_points);
        
        double dt = observation_time_days / num_points;
        
        for (int i = 0; i < num_points; ++i) {
            double time = i * dt;
            signature.time_points.push_back(time);
            
            // Calculate primary phase shift
            double phase_shift = calculatePhaseShift(system, time);
            
            // Add systematic effects
            double stellar_activity = 0.1 * std::sin(2.0 * M_PI * time / 25.0); // 25-day rotation
            double instrumental_drift = 1e-12 * time / observation_time_days;
            
            // Add noise (depends on system distance and planet mass)
            double base_noise = 1e-15 / (system.planet_mass * system.star_mass);
            double distance_factor = std::pow(system.distance_to_system / 100.0, 2.0);
            double noise_level = base_noise * distance_factor;
            double noise = noise_dist(rng) * noise_level;
            
            signature.phase_shifts.push_back(phase_shift + stellar_activity + instrumental_drift + noise);
            signature.noise_levels.push_back(noise_level);
            
            // Amplitude variations (gravitational lensing effects)
            double lensing_amplitude = 1e-6 * system.planet_mass / std::pow(system.distance_to_system, 2.0);
            double orbital_phase = 2.0 * M_PI * time / system.orbital_period + system.phase_offset;
            double amplitude_variation = lensing_amplitude * std::cos(orbital_phase);
            signature.amplitude_variations.push_back(amplitude_variation);
            
            // Complex gravitational field representation
            std::complex<double> field(phase_shift, amplitude_variation);
            signature.gravitational_field.push_back(field);
        }
        
        // Calculate detection metrics
        calculateDetectionMetrics(signature, system);
        
        return signature;
    }
    
    // Calculate detection SNR and confidence
    void calculateDetectionMetrics(GPISignature& signature, const PlanetarySystem& system) {
        // Calculate RMS of phase shifts
        double mean_phase = 0.0;
        for (double phase : signature.phase_shifts) {
            mean_phase += phase;
        }
        mean_phase /= signature.phase_shifts.size();
        
        double rms_signal = 0.0;
        for (double phase : signature.phase_shifts) {
            rms_signal += (phase - mean_phase) * (phase - mean_phase);
        }
        rms_signal = std::sqrt(rms_signal / signature.phase_shifts.size());
        
        // Calculate noise RMS
        double rms_noise = 0.0;
        for (double noise : signature.noise_levels) {
            rms_noise += noise * noise;
        }
        rms_noise = std::sqrt(rms_noise / signature.noise_levels.size());
        
        // SNR calculation
        signature.detection_snr = rms_signal / rms_noise;
        
        // Confidence level based on SNR and system parameters
        double mass_factor = std::log10(system.planet_mass);
        double distance_factor = 1.0 / std::log10(system.distance_to_system);
        double period_factor = 1.0 / std::sqrt(system.orbital_period);
        
        signature.confidence_level = std::tanh(
            signature.detection_snr * mass_factor * distance_factor * period_factor / 10.0
        );
        
        // Ensure confidence is between 0 and 1
        signature.confidence_level = std::max(0.0, std::min(1.0, signature.confidence_level));
    }
    
    // Export data to JSON format for Python integration
    std::string exportToJSON(const GPISignature& signature, const PlanetarySystem& system) {
        std::ostringstream json;
        json << "{\n";
        json << "  \"system_parameters\": {\n";
        json << "    \"star_mass\": " << system.star_mass << ",\n";
        json << "    \"star_radius\": " << system.star_radius << ",\n";
        json << "    \"planet_mass\": " << system.planet_mass << ",\n";
        json << "    \"planet_radius\": " << system.planet_radius << ",\n";
        json << "    \"orbital_period\": " << system.orbital_period << ",\n";
        json << "    \"semi_major_axis\": " << system.semi_major_axis << ",\n";
        json << "    \"eccentricity\": " << system.eccentricity << ",\n";
        json << "    \"inclination\": " << system.inclination << ",\n";
        json << "    \"distance_to_system\": " << system.distance_to_system << "\n";
        json << "  },\n";
        json << "  \"detection_metrics\": {\n";
        json << "    \"snr\": " << signature.detection_snr << ",\n";
        json << "    \"confidence\": " << signature.confidence_level << ",\n";
        json << "    \"data_points\": " << signature.time_points.size() << "\n";
        json << "  },\n";
        json << "  \"time_series\": {\n";
        json << "    \"time\": [";
        for (size_t i = 0; i < signature.time_points.size(); ++i) {
            if (i > 0) json << ", ";
            json << signature.time_points[i];
        }
        json << "],\n";
        json << "    \"phase_shifts\": [";
        for (size_t i = 0; i < signature.phase_shifts.size(); ++i) {
            if (i > 0) json << ", ";
            json << signature.phase_shifts[i];
        }
        json << "],\n";
        json << "    \"amplitude_variations\": [";
        for (size_t i = 0; i < signature.amplitude_variations.size(); ++i) {
            if (i > 0) json << ", ";
            json << signature.amplitude_variations[i];
        }
        json << "]\n";
        json << "  }\n";
        json << "}\n";
        
        return json.str();
    }
    
    // Generate multiple synthetic systems for testing
    std::vector<std::pair<PlanetarySystem, GPISignature>> generateTestDataset(int num_systems = 100) {
        std::vector<std::pair<PlanetarySystem, GPISignature>> dataset;
        dataset.reserve(num_systems);
        
        for (int i = 0; i < num_systems; ++i) {
            PlanetarySystem system = generateRealisticSystem();
            GPISignature signature = generateGPISignature(system, 1000, 365.0);
            dataset.emplace_back(system, signature);
        }
        
        return dataset;
    }
};

// C interface for Python integration
extern "C" {
    GPIDataGenerator* create_gpi_generator(unsigned int seed) {
        return new GPIDataGenerator(seed);
    }
    
    void destroy_gpi_generator(GPIDataGenerator* generator) {
        delete generator;
    }
    
    char* generate_gpi_data(GPIDataGenerator* generator, int num_points, double observation_time) {
        auto system = generator->generateRealisticSystem();
        auto signature = generator->generateGPISignature(system, num_points, observation_time);
        std::string json = generator->exportToJSON(signature, system);
        
        char* result = new char[json.length() + 1];
        std::strcpy(result, json.c_str());
        return result;
    }
    
    void free_string(char* str) {
        delete[] str;
    }
}

// Main function for standalone testing
int main() {
    std::cout << "GPI Data Generator - High-Precision Synthetic Data\n";
    std::cout << "================================================\n\n";
    
    GPIDataGenerator generator;
    
    // Generate a test system
    auto system = generator.generateRealisticSystem();
    std::cout << "Generated planetary system:\n";
    std::cout << "  Star mass: " << system.star_mass << " solar masses\n";
    std::cout << "  Planet mass: " << system.planet_mass << " Earth masses\n";
    std::cout << "  Orbital period: " << system.orbital_period << " days\n";
    std::cout << "  Distance: " << system.distance_to_system << " parsecs\n\n";
    
    // Generate GPI signature
    auto signature = generator.generateGPISignature(system, 1000, 365.0);
    std::cout << "GPI signature generated:\n";
    std::cout << "  Data points: " << signature.time_points.size() << "\n";
    std::cout << "  Detection SNR: " << signature.detection_snr << "\n";
    std::cout << "  Confidence level: " << signature.confidence_level << "\n\n";
    
    // Export to file
    std::ofstream outfile("gpi_test_data.json");
    outfile << generator.exportToJSON(signature, system);
    outfile.close();
    
    std::cout << "Data exported to gpi_test_data.json\n";
    
    return 0;
}
