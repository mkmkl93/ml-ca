#include <algorithm>
#include <iterator>
#include <fstream>
#include <random>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <stack>

#define MAX_GY 10.0
#define MIN_GY 0.75
#define GRANULATION_GY 0.125
#define MIN_RADIATION_GY 0.5
struct DataFrame {
  float sum;
  std::vector<float> coins;
  std::vector<float> avail_coins;
};

typedef std::map<float, std::vector<std::vector<float>>> protocol_sum;

int main() {
  	protocol_sum protocol_map;
	int ways = 0;
//	int oldWays = 0;
	for (float target = MAX_GY; target >= MIN_GY; target -= GRANULATION_GY) {
		std::stack<DataFrame> s;
		std::vector<float> max_doses;
		for (float max_dose = MIN_RADIATION_GY; max_dose <= MAX_GY; max_dose += GRANULATION_GY) {
			max_doses.push_back(max_dose);
		}
		s.push({ target, {}, max_doses });
		std::vector<std::vector<float>> chosen_protocols;
		while (!s.empty()) {
			DataFrame top = s.top();
			s.pop();
			if (top.sum < 0) continue;

			if (top.sum == 0) {
				chosen_protocols.push_back(top.coins);
				++ways;
				continue;
			}

			if (top.avail_coins.empty()) continue;
			DataFrame d = top;
			d.sum -= top.avail_coins.back();
			d.coins.push_back(top.avail_coins.back());
			s.push(d);
			d = top;
			d.avail_coins.pop_back();
			s.push(d);
		}
//		std::cout << target << ":" << ways << " " << ways - oldWays << "\n";
//		oldWays = ways;
		protocol_map[target] = chosen_protocols;
	}
	std::cout << ways << std::endl;
	std::cout << "Size " << protocol_map.size() << "\n";

	// 1. Losowanko
	// 1.1 losuj sumę - równa dystrybucja po wartości sumy
	// 1.2 losuj pozycję - czyli losowy rozkład dawek o danej sumie
	// 1.3 losuj permutacje
	// 1.4 losuj czasy

	std::set<std::vector<float>> protocol_set;

	std::random_device rd;
	std::mt19937 rng;
	rng.seed(42);
	std::uniform_int_distribution<int> dist_sum(MIN_GY / GRANULATION_GY, MAX_GY / GRANULATION_GY);
	int i = 0;
    std::ofstream f("../data/old_data/protocols/protocol_list_1.csv");
    std::ofstream t("../data/old_data/protocols/protocol_times_1.csv");
    int file_count = 1;
    // Liczba protokołów
    while (i < 50000) {
        // Ile linii na jeden plik: 50000 / 32 = 1562.5 ~ 1570,
        // po przekroczeniu tego otwieramy kolejny plik
        if (i > file_count * 1570) {
            file_count++;
            f.close();
            t.close();
            f.open("../data/old_data/protocols/protocol_list_" + std::to_string(file_count) + ".csv");
            t.open("../data/old_data/protocols/protocol_times_" + std::to_string(file_count) + ".csv");
        }
        // Losuj sumę - od 0.75 do 10.00
        float sum = dist_sum(rng) * GRANULATION_GY;

        std::vector<float> temp_protocol;

        std::cout << "Suma " << sum << "\n";
        std::cout << "Rozmiar " << protocol_map[sum].size() << "\n";
//		 Losuj pozycję
        std::uniform_int_distribution<int> dist_pos(0, protocol_map[sum].size() - 1);
        int temp_pos = dist_pos(rng);
        std::cout << "Poz " << temp_pos << "\n";

        temp_protocol = (protocol_map[sum])[temp_pos];

        int temp_protocol_length = temp_protocol.size();

        //std::cout << "Dlugosc " << temp_protocol_length << "\n";

        std::sort(temp_protocol.begin(), temp_protocol.begin() + temp_protocol_length);

        // Losuj permutację
        std::random_shuffle ( temp_protocol.begin(), temp_protocol.end() );

        // Zapisz jeśli nie było powtórek
        if (protocol_set.find(temp_protocol) == protocol_set.end()) {
            std::cout << "Zapis.\n";
            protocol_set.insert(temp_protocol);
            i++;
            std::cout << i << "...\n";

            // Losuj 2 czasy
            // _doses_slots_ kroków symulacji co 6 sekund
            // zaczynamy w pierwszym kroku nr 1
            // między dawkami 10 min = _doses_min_interval_ kroków
            // 1.1 losuj aż nie będzie odpowiedniej długości

            std::vector<int> random_time_1;
            std::vector<int> random_time_2;

            int time_length = 0;

            // 72000 = 5 dni co 6 sekund, interwał 100, dzień 14400, 8h to 4800
            // 720 = 5 dni co 10 minut, interwał 1, dzień 144, 6h to 36
            int doses_slots = 720;
            int doses_multiply = 72000 / doses_slots;
            int doses_min_interval = 3;
//
            while (time_length < temp_protocol_length) {
                std::uniform_int_distribution<int> dist_time(1, doses_slots);
                int temp_time = dist_time(rng);
                bool good_time = true;
                for (int t = 0; t < (int)random_time_1.size(); ++t) {
                    if ((std::abs(random_time_1[t] - doses_multiply*temp_time)) < doses_multiply*doses_min_interval) {
                        good_time = false;
                    }
                }
                if (good_time) {
                    random_time_1.push_back(doses_multiply*temp_time);
                    time_length++;
                }
            }
//
            time_length = 0;

            while (time_length < temp_protocol_length) {
                std::uniform_int_distribution<int> dist_time(1, doses_slots);
                int temp_time = dist_time(rng);
                bool good_time = true;
                for (int t = 0; t < (int)random_time_2.size(); ++t) {
                    if ((std::abs(random_time_2[t] - doses_multiply*temp_time)) < doses_multiply*doses_min_interval) {
                        good_time = false;
                    }
                }
                if (good_time) {
                    random_time_2.push_back(doses_multiply*temp_time);
                    time_length++;
                }
            }

            std::cout << "Good times.\n";

            std::sort(random_time_1.begin(), random_time_1.begin() + temp_protocol_length);
            std::sort(random_time_2.begin(), random_time_2.begin() + temp_protocol_length);



            for (std::vector<float>::const_iterator val = temp_protocol.begin();
                 val != temp_protocol.end(); ++val) {
                t << *val << ' ';
            }
            t << '\n';
            for (std::vector<int>::const_iterator val = random_time_1.begin();
                 val != random_time_1.end(); ++val) {
                f << *val << ' ';
                t << *val << ' ';
            }
            f << '\n';
            t << '\n';
            for (std::vector<float>::const_iterator val = temp_protocol.begin();
                 val != temp_protocol.end(); ++val) {
                t << *val << ' ';
            }
            t << '\n';
            for (std::vector<int>::const_iterator val = random_time_2.begin();
                 val != random_time_2.end(); ++val) {
                f << *val << ' ';
                t << *val << ' ';
            }
            f << '\n';
            t << '\n';
        }
    }
    return 0;
}
