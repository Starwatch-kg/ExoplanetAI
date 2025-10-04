/*
Search Accelerator - C++ module for high-performance exoplanet search algorithms
Optimized implementations of BLS, GPI, and other detection methods
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <omp.h>
#include <chrono>
#include <memory>
#include <unordered_map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class SearchAccelerator {
private:
    // FFTW plans for efficient FFT computation
    fftw_plan fft_forward_plan;
    fftw_plan fft_backward_plan;
    fftw_complex* fft_input;
    fftw_complex* fft_output;
    int fft_size;
    
    // OpenMP thread count
    int num_threads;
    
public:
    SearchAccelerator(int max_fft_size = 65536, int threads = 0) : 
        fft_size(max_fft_size),
        num_threads(threads > 0 ? threads : omp_get_max_threads()) {
        
        // Initialize FFTW
        fftw_init_threads();
        fftw_plan_with_nthreads(num_threads);
        
        // Allocate FFT arrays
        fft_input = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_size);
        fft_output = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_size);
        
        // Create FFTW plans
        fft_forward_plan = fftw_plan_dft_1d(fft_size, fft_input, fft_output, FFTW_FORWARD, FFTW_MEASURE);
        fft_backward_plan = fftw_plan_dft_1d(fft_size, fft_output, fft_input, FFTW_BACKWARD, FFTW_MEASURE);
        
        std::cout << "Search Accelerator initialized with " << num_threads << " threads\n";
    }
    
    ~SearchAccelerator() {
        fftw_destroy_plan(fft_forward_plan);
        fftw_destroy_plan(fft_backward_plan);
        fftw_free(fft_input);
        fftw_free(fft_output);
        fftw_cleanup_threads();
    }
    
    struct TimeSeriesData {
        std::vector<double> time;
        std::vector<double> flux;
        std::vector<double> flux_err;
        double cadence;
        double duration;
    };
    
    struct BLSResult {
        double period;
        double epoch;
        double duration;
        double depth;
        double snr;
        double power;
        std::vector<double> folded_time;
        std::vector<double> folded_flux;
        std::vector<double> period_grid;
        std::vector<double> power_spectrum;
    };
    
    struct GPIResult {
        double detection_confidence;
        double phase_amplitude;
        double orbital_period;
        std::vector<double> phase_shifts;
        std::vector<double> gravitational_signature;
        double snr;
    };
    
    // Optimized Box Least Squares implementation
    BLSResult acceleratedBLS(const TimeSeriesData& data, 
                           double period_min = 0.5, 
                           double period_max = 50.0,
                           int period_samples = 10000,
                           double duration_factor = 0.1) {
        
        BLSResult result;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Prepare data
        int n_data = data.time.size();
        std::vector<double> normalized_flux(n_data);
        
        // Normalize flux
        double mean_flux = 0.0;
        #pragma omp parallel for reduction(+:mean_flux)
        for (int i = 0; i < n_data; ++i) {
            mean_flux += data.flux[i];
        }
        mean_flux /= n_data;
        
        #pragma omp parallel for
        for (int i = 0; i < n_data; ++i) {
            normalized_flux[i] = (data.flux[i] - mean_flux) / mean_flux;
        }
        
        // Generate period grid
        result.period_grid.resize(period_samples);
        result.power_spectrum.resize(period_samples);
        
        double log_period_min = std::log(period_min);
        double log_period_max = std::log(period_max);
        double log_period_step = (log_period_max - log_period_min) / (period_samples - 1);
        
        #pragma omp parallel for
        for (int i = 0; i < period_samples; ++i) {
            result.period_grid[i] = std::exp(log_period_min + i * log_period_step);
        }
        
        // BLS search
        double max_power = 0.0;
        int best_period_idx = 0;
        
        #pragma omp parallel for
        for (int p = 0; p < period_samples; ++p) {
            double period = result.period_grid[p];
            double max_duration = period * duration_factor;
            
            // Phase fold the data
            std::vector<std::pair<double, double>> folded_data(n_data);
            for (int i = 0; i < n_data; ++i) {
                double phase = std::fmod(data.time[i], period) / period;
                folded_data[i] = {phase, normalized_flux[i]};
            }
            
            // Sort by phase
            std::sort(folded_data.begin(), folded_data.end());
            
            // Search for best transit duration and epoch
            double best_power_for_period = 0.0;
            
            for (int epoch_idx = 0; epoch_idx < n_data / 10; ++epoch_idx) {
                double epoch_phase = static_cast<double>(epoch_idx) / (n_data / 10);
                
                for (double duration = period * 0.01; duration <= max_duration; duration *= 1.1) {
                    double duration_phase = duration / period;
                    
                    // Calculate BLS statistic
                    double in_transit_sum = 0.0;
                    double out_transit_sum = 0.0;
                    int in_transit_count = 0;
                    int out_transit_count = 0;
                    
                    for (const auto& point : folded_data) {
                        double phase_diff = std::abs(point.first - epoch_phase);
                        if (phase_diff > 0.5) phase_diff = 1.0 - phase_diff;
                        
                        if (phase_diff <= duration_phase / 2.0) {
                            in_transit_sum += point.second;
                            in_transit_count++;
                        } else {
                            out_transit_sum += point.second;
                            out_transit_count++;
                        }
                    }
                    
                    if (in_transit_count > 0 && out_transit_count > 0) {
                        double in_transit_mean = in_transit_sum / in_transit_count;
                        double out_transit_mean = out_transit_sum / out_transit_count;
                        double depth = out_transit_mean - in_transit_mean;
                        
                        // BLS power calculation
                        double power = depth * depth * in_transit_count * out_transit_count / 
                                     (in_transit_count + out_transit_count);
                        
                        if (power > best_power_for_period) {
                            best_power_for_period = power;
                        }
                    }
                }
            }
            
            result.power_spectrum[p] = best_power_for_period;
            
            #pragma omp critical
            {
                if (best_power_for_period > max_power) {
                    max_power = best_power_for_period;
                    best_period_idx = p;
                }
            }
        }
        
        // Extract best parameters
        result.period = result.period_grid[best_period_idx];
        result.power = max_power;
        
        // Calculate SNR
        double mean_power = 0.0;
        double std_power = 0.0;
        
        for (double power : result.power_spectrum) {
            mean_power += power;
        }
        mean_power /= result.power_spectrum.size();
        
        for (double power : result.power_spectrum) {
            std_power += (power - mean_power) * (power - mean_power);
        }
        std_power = std::sqrt(std_power / result.power_spectrum.size());
        
        result.snr = (result.power - mean_power) / std_power;
        
        // Generate folded light curve for best period
        generateFoldedLightCurve(data, result);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "BLS search completed in " << duration_ms.count() << " ms\n";
        std::cout << "Best period: " << result.period << " days, SNR: " << result.snr << "\n";
        
        return result;
    }
    
    // Optimized GPI analysis
    GPIResult acceleratedGPI(const TimeSeriesData& data, 
                           double phase_sensitivity = 1e-12,
                           double period_min = 0.1,
                           double period_max = 1000.0) {
        
        GPIResult result;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int n_data = data.time.size();
        
        // Calculate phase shifts using gravitational field analysis
        result.phase_shifts.resize(n_data);
        result.gravitational_signature.resize(n_data);
        
        // Apply Savitzky-Golay filter for noise reduction
        std::vector<double> filtered_flux = applySavitzkyGolayFilter(data.flux, 5, 2);
        
        // Calculate differential phase shifts
        #pragma omp parallel for
        for (int i = 1; i < n_data - 1; ++i) {
            double dt = data.time[i+1] - data.time[i-1];
            double dflux = filtered_flux[i+1] - filtered_flux[i-1];
            
            // Gravitational phase shift calculation
            double phase_shift = phase_sensitivity * dflux / dt;
            result.phase_shifts[i] = phase_shift;
            
            // Gravitational signature (second derivative)
            double d2flux = filtered_flux[i+1] - 2*filtered_flux[i] + filtered_flux[i-1];
            result.gravitational_signature[i] = d2flux / (dt * dt);
        }
        
        // Boundary conditions
        result.phase_shifts[0] = result.phase_shifts[1];
        result.phase_shifts[n_data-1] = result.phase_shifts[n_data-2];
        result.gravitational_signature[0] = result.gravitational_signature[1];
        result.gravitational_signature[n_data-1] = result.gravitational_signature[n_data-2];
        
        // Frequency domain analysis using FFT
        if (n_data <= fft_size) {
            // Prepare FFT input
            for (int i = 0; i < n_data; ++i) {
                fft_input[i][0] = result.phase_shifts[i];
                fft_input[i][1] = 0.0;
            }
            for (int i = n_data; i < fft_size; ++i) {
                fft_input[i][0] = 0.0;
                fft_input[i][1] = 0.0;
            }
            
            // Execute FFT
            fftw_execute(fft_forward_plan);
            
            // Find dominant frequency
            double max_amplitude = 0.0;
            int max_freq_idx = 0;
            
            for (int i = 1; i < fft_size / 2; ++i) {
                double amplitude = std::sqrt(fft_output[i][0] * fft_output[i][0] + 
                                           fft_output[i][1] * fft_output[i][1]);
                if (amplitude > max_amplitude) {
                    max_amplitude = amplitude;
                    max_freq_idx = i;
                }
            }
            
            // Convert frequency to period
            double frequency = static_cast<double>(max_freq_idx) / (data.duration);
            result.orbital_period = 1.0 / frequency;
            result.phase_amplitude = max_amplitude;
            
            // Clamp period to reasonable range
            result.orbital_period = std::max(period_min, std::min(period_max, result.orbital_period));
        }
        
        // Calculate detection confidence
        calculateGPIConfidence(result, data);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "GPI analysis completed in " << duration_ms.count() << " ms\n";
        std::cout << "Detection confidence: " << result.detection_confidence << "\n";
        std::cout << "Orbital period: " << result.orbital_period << " days\n";
        
        return result;
    }
    
    // Parallel period search for multiple targets
    std::vector<BLSResult> batchBLSSearch(const std::vector<TimeSeriesData>& datasets,
                                        double period_min = 0.5,
                                        double period_max = 50.0) {
        
        std::vector<BLSResult> results(datasets.size());
        
        #pragma omp parallel for
        for (size_t i = 0; i < datasets.size(); ++i) {
            results[i] = acceleratedBLS(datasets[i], period_min, period_max);
        }
        
        return results;
    }
    
private:
    void generateFoldedLightCurve(const TimeSeriesData& data, BLSResult& result) {
        int n_data = data.time.size();
        result.folded_time.resize(n_data);
        result.folded_flux.resize(n_data);
        
        for (int i = 0; i < n_data; ++i) {
            result.folded_time[i] = std::fmod(data.time[i], result.period) / result.period;
            result.folded_flux[i] = data.flux[i];
        }
        
        // Sort by folded time
        std::vector<std::pair<double, double>> folded_pairs(n_data);
        for (int i = 0; i < n_data; ++i) {
            folded_pairs[i] = {result.folded_time[i], result.folded_flux[i]};
        }
        
        std::sort(folded_pairs.begin(), folded_pairs.end());
        
        for (int i = 0; i < n_data; ++i) {
            result.folded_time[i] = folded_pairs[i].first;
            result.folded_flux[i] = folded_pairs[i].second;
        }
    }
    
    std::vector<double> applySavitzkyGolayFilter(const std::vector<double>& data, 
                                               int window_size, int poly_order) {
        std::vector<double> filtered(data.size());
        int half_window = window_size / 2;
        
        // Simple moving average for now (can be replaced with full SG filter)
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); ++i) {
            double sum = 0.0;
            int count = 0;
            
            for (int j = -half_window; j <= half_window; ++j) {
                int idx = static_cast<int>(i) + j;
                if (idx >= 0 && idx < static_cast<int>(data.size())) {
                    sum += data[idx];
                    count++;
                }
            }
            
            filtered[i] = sum / count;
        }
        
        return filtered;
    }
    
    void calculateGPIConfidence(GPIResult& result, const TimeSeriesData& data) {
        // Calculate RMS of phase shifts
        double rms_phase = 0.0;
        for (double phase : result.phase_shifts) {
            rms_phase += phase * phase;
        }
        rms_phase = std::sqrt(rms_phase / result.phase_shifts.size());
        
        // Estimate noise level
        double noise_level = 1e-15; // Typical GPI noise level
        if (!data.flux_err.empty()) {
            double mean_err = 0.0;
            for (double err : data.flux_err) {
                mean_err += err;
            }
            noise_level = mean_err / data.flux_err.size();
        }
        
        // Calculate SNR
        result.snr = rms_phase / noise_level;
        
        // Confidence based on SNR and signal characteristics
        double snr_factor = std::tanh(result.snr / 5.0);
        double amplitude_factor = std::tanh(result.phase_amplitude * 1e12);
        double period_factor = 1.0 / (1.0 + std::exp(-(result.orbital_period - 10.0) / 5.0));
        
        result.detection_confidence = snr_factor * amplitude_factor * period_factor;
        result.detection_confidence = std::max(0.0, std::min(1.0, result.detection_confidence));
    }
};

// C interface for Python integration
extern "C" {
    SearchAccelerator* create_search_accelerator(int max_fft_size, int threads) {
        return new SearchAccelerator(max_fft_size, threads);
    }
    
    void destroy_search_accelerator(SearchAccelerator* accelerator) {
        delete accelerator;
    }
    
    // BLS search interface
    void accelerated_bls_search(SearchAccelerator* accelerator,
                              double* time, double* flux, double* flux_err, int n_data,
                              double period_min, double period_max, int period_samples,
                              double* result_period, double* result_snr, double* result_power) {
        
        SearchAccelerator::TimeSeriesData data;
        data.time.assign(time, time + n_data);
        data.flux.assign(flux, flux + n_data);
        data.flux_err.assign(flux_err, flux_err + n_data);
        data.cadence = (time[n_data-1] - time[0]) / n_data;
        data.duration = time[n_data-1] - time[0];
        
        auto result = accelerator->acceleratedBLS(data, period_min, period_max, period_samples);
        
        *result_period = result.period;
        *result_snr = result.snr;
        *result_power = result.power;
    }
    
    // GPI analysis interface
    void accelerated_gpi_analysis(SearchAccelerator* accelerator,
                                double* time, double* flux, double* flux_err, int n_data,
                                double phase_sensitivity,
                                double* result_confidence, double* result_period, double* result_snr) {
        
        SearchAccelerator::TimeSeriesData data;
        data.time.assign(time, time + n_data);
        data.flux.assign(flux, flux + n_data);
        data.flux_err.assign(flux_err, flux_err + n_data);
        data.cadence = (time[n_data-1] - time[0]) / n_data;
        data.duration = time[n_data-1] - time[0];
        
        auto result = accelerator->acceleratedGPI(data, phase_sensitivity);
        
        *result_confidence = result.detection_confidence;
        *result_period = result.orbital_period;
        *result_snr = result.snr;
    }
}

// Main function for standalone testing
int main() {
    std::cout << "Search Accelerator - High-Performance Exoplanet Detection\n";
    std::cout << "======================================================\n\n";
    
    SearchAccelerator accelerator;
    
    // Generate test data
    SearchAccelerator::TimeSeriesData test_data;
    int n_points = 10000;
    double period = 3.5; // days
    double depth = 0.01; // 1% transit depth
    double noise_level = 0.001;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noise_level);
    
    for (int i = 0; i < n_points; ++i) {
        double time = i * 0.02; // 20-minute cadence
        double flux = 1.0;
        
        // Add transit signal
        double phase = std::fmod(time, period) / period;
        if (phase > 0.48 && phase < 0.52) { // 4% transit duration
            flux -= depth;
        }
        
        // Add noise
        flux += noise(gen);
        
        test_data.time.push_back(time);
        test_data.flux.push_back(flux);
        test_data.flux_err.push_back(noise_level);
    }
    
    test_data.cadence = 0.02;
    test_data.duration = test_data.time.back() - test_data.time.front();
    
    std::cout << "Generated test data: " << n_points << " points over " 
              << test_data.duration << " days\n\n";
    
    // Test BLS
    std::cout << "Running accelerated BLS search...\n";
    auto bls_result = accelerator.acceleratedBLS(test_data, 1.0, 10.0, 1000);
    
    std::cout << "\nBLS Results:\n";
    std::cout << "  Detected period: " << bls_result.period << " days (true: " << period << ")\n";
    std::cout << "  SNR: " << bls_result.snr << "\n";
    std::cout << "  Power: " << bls_result.power << "\n\n";
    
    // Test GPI
    std::cout << "Running accelerated GPI analysis...\n";
    auto gpi_result = accelerator.acceleratedGPI(test_data);
    
    std::cout << "\nGPI Results:\n";
    std::cout << "  Detection confidence: " << gpi_result.detection_confidence << "\n";
    std::cout << "  Orbital period: " << gpi_result.orbital_period << " days\n";
    std::cout << "  SNR: " << gpi_result.snr << "\n";
    
    return 0;
}
