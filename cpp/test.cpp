// Example program
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cassert>

class Polygon {
    int s;
    std::shared_ptr<int> map;
    bool transformed;
    public:
        Polygon(int s) 
        {
            this->s = s;
            this->transformed = false;
        }
        
        void set_s(int s) 
        {
            this->transformed = true;
            this->s=s;
        }
        
        void recalculate_map()
        {
            std::cout << "recalculating map\n";
            this->map = std::make_shared<int>(2*this->s);
        }
        
        int get_map() {
            if (this->transformed) {
                this->recalculate_map();
            } else {
                if (this->map) {
                    std::cout << "map exists\n";
                } else {
                    std::cout << "map doesnt exist\n";
                    this->recalculate_map();
                }
            }
            return *this->map;
        }
};

int main()
{
    // Polygon p(5);
    // std::cout << p.get_map() << std::endl;
    // std::cout << p.get_map() << std::endl;
    // p.set_s(10);
    // std::cout << p.get_map() << std::endl;
    std::vector<int> size{512, 512}, hw;
    int scale_factor = 4;
    assert(size[0]%scale_factor==0);
    assert(size[1]%scale_factor==0);
    hw.push_back(5/2);
    hw = {1,2,3,4,5};

    std::cout << "The vector elements are: "; 
    // for (auto it = hw.begin(); it != hw.end(); ++it) 
    //     std::cout << *it << " "; 
    for(auto const& value: hw) {
        std::cout << value*2;
    }
    return 0;
}