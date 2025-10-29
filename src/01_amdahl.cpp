#include <bits/stdc++.h>

using namespace std;

int main() {

    //fraction of code that can be ||ised
    double p;

    //processors
    int n;


    cin >> p;
    cin >> n;

    double speedup = 1.0 / ((1 - p) + (p/n));

    double maxSpeedup = 1.0 / ((1 - p));


    cout << "speedup = " << speedup << endl;
    cout << "theoritical max speedup = " << maxSpeedup << endl;



    return 0;
}