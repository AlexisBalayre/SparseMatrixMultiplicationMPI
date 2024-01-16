#include <iostream>
#include <vector>

// Fonction pour multiplier une matrice creuse par un vecteur dense
std::vector<double> sparseMatrixVectorMultiply(const std::vector<double> &values,
                                               const std::vector<int> &rows,
                                               const std::vector<int> &cols,
                                               const std::vector<double> &vecteur,
                                               int numRows)
{
    std::vector<double> result(numRows, 0.0);

    for (int i = 0; i < numRows; ++i)
    {
        for (int j = rows[i]; j < rows[i + 1]; ++j)
        {
            result[i] += values[j] * vecteur[cols[j]];
        }
    }

    return result;
}

int main()
{
    // Exemple de matrice creuse au format CSR
    std::vector<double> values = {1, 2, 3, 4}; // Valeurs non-nulles
    std::vector<int> rows = {0, 2, 3, 3, 4};   // Index de début de chaque ligne + 1 après la dernière valeur
    std::vector<int> cols = {0, 2, 2, 3};      // Indices de colonne pour chaque valeur non-nulle

    // Vecteur dense
    std::vector<double> vecteur = {1, 2, 3, 4};

    // Multiplication
    std::vector<double> resultat = sparseMatrixVectorMultiply(values, rows, cols, vecteur, 4);

    // Afficher le résultat
    for (double val : resultat)
    {
        std::cout << val << " ";
    }

    return 0;
}
