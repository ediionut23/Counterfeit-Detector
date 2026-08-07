#include <iostream>
#include <cstring>

using namespace std;

int main()
{
    char s1[31], s2[31], aux[31];
    bool gasit = false;

    cin >> s1 >> s2;

    int n1 = strlen(s1);
    int n2 = strlen(s2);

    for (int i = 0; i < n1; i++)
    {
        int lungimeSufix = n1 - i;

        if (lungimeSufix <= n2)
        {

            strncpy(aux, s2, lungimeSufix);
            aux[lungimeSufix] = '\0';

            if (strcmp(s1 + i, aux) == 0)
            {
                cout << s1 + i << " ";
                gasit = true;
            }
        }
    }

    if (!gasit)
    {
        cout << "NU EXISTA";
    }

    return 0;
}