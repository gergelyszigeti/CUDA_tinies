class MyRand {
public:
    MyRand(int init_state = 19790327) : rand_state(init_state) {}
    unsigned int myrand();
    unsigned int operator()() { return myrand(); }

private:
    unsigned int rand_state;
};

unsigned int MyRand::myrand()
{
    rand_state *= 65327;      // the day when I was exactly -14 years old (my birthday is not prime :( )
    rand_state ^= rand_state >> 5;
    return rand_state;
}

