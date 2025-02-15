//
// Created by piotro on 15.02.25.
//
// File: ./test/holorunner_test.cpp

#include "gtest/gtest.h"
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>  // for std::remove
#include <boost/tokenizer.hpp>

//------------------------------------------------------------------------------
// get_file_contents
// Copied from holorunner.cpp for testing purposes.
//------------------------------------------------------------------------------
std::string get_file_contents(const char* filename) {
    std::string contents, temp;
    std::fstream input(filename, std::ios::in);
    if (input.is_open()) {
        while (std::getline(input, temp)) {
            temp.erase(std::remove_if(temp.begin(),
                                        temp.end(),
                                        [](char c) { return (c == '\r' || c == '\t' || c == ' ' || c == '\n'); }),
                       temp.end());
            contents += temp;
        }
    }
    return contents;
}

//------------------------------------------------------------------------------
// Test: get_file_contents removes all whitespace from a file.
//------------------------------------------------------------------------------
TEST(HolorunnerTest, GetFileContentsRemovesWhitespace) {
    // Create a temporary file with known content.
    std::string filename = "temp_test_mask.txt";
    std::ofstream ofs(filename);
    ASSERT_TRUE(ofs.is_open());
    // Include various whitespace characters.
    ofs << " image1.jpg \n; image2.jpg\t;image3.jpg \r\n";
    ofs.close();

    std::string contents = get_file_contents(filename.c_str());
    // Expect that the returned string is the concatenation of tokens with no whitespace.
    EXPECT_EQ(contents, "image1.jpg;image2.jpg;image3.jpg");

    // Remove the temporary file.
    std::remove(filename.c_str());
}

//------------------------------------------------------------------------------
// Test: Tokenization using boost::tokenizer splits a semicolon-separated string.
//------------------------------------------------------------------------------
TEST(HolorunnerTest, TokenizationTest) {
    std::string pixel_setup = "image1.jpg;image2.jpg;image3.jpg";
    typedef boost::char_separator<char> separator_type;
    boost::tokenizer<separator_type> tokens(pixel_setup, separator_type(";"));

    std::vector<std::string> tokenList;
    for (const auto& token : tokens) {
        tokenList.push_back(token);
    }
    EXPECT_EQ(tokenList.size(), 3);
    EXPECT_EQ(tokenList[0], "image1.jpg");
    EXPECT_EQ(tokenList[1], "image2.jpg");
    EXPECT_EQ(tokenList[2], "image3.jpg");
}

//------------------------------------------------------------------------------
// Test: Calculation of delay in milliseconds from frequency.
// For a frequency of 10 Hz, delay should be 100 ms.
//------------------------------------------------------------------------------
TEST(HolorunnerTest, MsDelayCalculation) {
    float frequency = 10.0f;
    int ms_delay = static_cast<int>(1000.0f / frequency);
    EXPECT_EQ(ms_delay, 100);
}
