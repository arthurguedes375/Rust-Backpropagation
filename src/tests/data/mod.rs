use na::{DMatrix, dmatrix};

pub fn forwarded_values() -> (Vec<DMatrix<f32>>, Vec<DMatrix<f32>>) {
    let z: Vec<DMatrix<f32>> = vec![
        dmatrix![
            -1.339008,  0.5912351, -0.46961677;
            -1.2453202,  0.5554719, -0.42961967;
            -1.1470667,  0.48611996, -0.41319478;
            -1.7281084,  0.83985245, -0.55552095;
            -1.1483712,  0.4957167, -0.40645987;
            -1.0566401,  0.4743486,  -0.3563604;
            -1.7304565,  0.8571266,  -0.5433981;
            -1.0565097,  0.47338894,  -0.3570339;
            -0.6727575,  0.26411825, -0.24351653;
            -0.9585171,  0.40595636, -0.33926207;
        ],
        dmatrix![
            0.02410519,  0.07107477, 0.023049207;
            0.023722487, 0.072783165, 0.023729634;
            0.023513988,  0.07486805, 0.023521315;
            0.024553824,  0.06369527, 0.023575902;
            0.02345122, 0.074770145, 0.023789676;
            0.022886634,  0.07640556, 0.024967542;
            0.024452163,  0.06355375, 0.024022019;
            0.022892859,  0.07641537, 0.024940608;
            0.02102612, 0.084577605, 0.026598945;
            0.02258544,  0.07856078,  0.02486846;
        ],
        dmatrix![
            -0.09025681,   0.099101886,  -0.024840042,    0.10654163, -0.0022801142,   -0.15961869,    -0.0370386,   0.041033063,    0.10336654,    0.07519531;
            -0.090273924,    0.09910561,  -0.024850223,    0.10650273, -0.0023252536,   -0.15960893,   -0.03702029,    0.04101605,   0.103378445,   0.075245745;
            -0.09032035,    0.09909926,  -0.024865363,     0.1064491, -0.0023698937,   -0.15959258,  -0.037003376,   0.040996626,    0.10338612,     0.0752821;
            -0.0900972,   0.099114716,  -0.024792476,    0.10672429, -0.0021225382,    -0.1596675,  -0.037096765,   0.041097853,    0.10333791,    0.07506181;
            -0.090311795,   0.099102095,  -0.024864107,   0.106453024, -0.0023704525,   -0.15959427,   -0.03700275,   0.040997107,    0.10338746,   0.075286664;
            -0.09031517,    0.09910974,  -0.024873547,    0.10641774, -0.0024194587,   -0.15958557,  -0.036981918,   0.040979315,    0.10340233,    0.07534778;
            -0.09008342,   0.099119514,  -0.024790453,    0.10673036,  -0.002123909,   -0.15967028,   -0.03709559,   0.041098524,    0.10334024,    0.07506984;
            -0.09031602,   0.099109456,  -0.024873674,   0.106417365, -0.0024193954,    -0.1595854,  -0.036981992,   0.040979274,   0.103402205,    0.07534731;
            -0.09043635,   0.099101126,  -0.024933401,    0.10621457, -0.0026222207,   -0.15952164,   -0.03689929,   0.040894695,   0.103448465,     0.0755499;
            -0.09036021,    0.09910317,  -0.024889795,    0.10636203,  -0.002467176,   -0.15956783,   -0.03696337,   0.040958397,    0.10341101,    0.07538819;
        ]
    ];

    let layers: Vec<DMatrix<f32>> = vec![
        dmatrix![
            1.0f32,  1.7,   15.0,    0.0;
            1.0,  1.5,   14.0,    0.0;
            1.0, 1.65,   13.0,    0.0;
            1.0,  1.4,   19.0,    0.0;
            1.0, 1.55,   13.0,    0.0;
            1.0,  1.2,   12.0,    0.0;
            1.0, 1.22,   19.0,    0.0;
            1.0, 1.21,   12.0,    0.0;
            1.0,  1.1,    8.0,    0.0;
            1.0, 1.34,   11.0,    0.0;
        ],
        dmatrix![
            1.0, 0.20767324,  0.6436485, 0.38470694;
            1.0,  0.2235113, 0.63540417, 0.39421713;
            1.0, 0.24102527,   0.619192,  0.3981463;
            1.0,  0.1508297,  0.6984341, 0.36458445;
            1.0, 0.24078673,  0.6214522, 0.39976126;
            1.0, 0.25795206,  0.6164125, 0.41184092;
            1.0,  0.1505292,  0.7020599, 0.36739746;
            1.0, 0.25797704, 0.61618555, 0.41167778;
            1.0,  0.3378797,  0.5656484, 0.43941996;
            1.0, 0.27717522,  0.6001179, 0.41598877;
        ],
        dmatrix![
            1.0,   0.506026,  0.5177612, 0.50576204;
            1.0, 0.50593036, 0.51818776,  0.5059321;
            1.0,  0.5058782,  0.5187083, 0.50588006;
            1.0, 0.50613815, 0.51591843,  0.5058937;
            1.0, 0.50586253,  0.5186838,  0.5059472;
            1.0,  0.5057214, 0.51909214, 0.50624156;
            1.0, 0.50611275,  0.5158831,  0.5060052;
            1.0,   0.505723,  0.5190945,  0.5062348;
            1.0,  0.5052563,  0.5211318,  0.5066493;
            1.0,  0.5056461,  0.5196301, 0.50621676;
        ],
        dmatrix![
            0.47745112, 0.52475524, 0.49379027, 0.52661026,    0.49943, 0.46017984,  0.4907414,  0.5102568, 0.52581865,    0.51879;
            0.47744682, 0.52475613,  0.4937877, 0.52660054,  0.4994187,  0.4601823,   0.490746, 0.51025254,  0.5258216,  0.5188026;
            0.47743526,  0.5247546,   0.493784,  0.5265871, 0.49940753, 0.46018633, 0.49075025,  0.5102477, 0.52582353, 0.51881164;
            0.4774909,  0.5247584,  0.4938022,  0.5266558, 0.49946937,  0.4601677,  0.4907269, 0.51027304,  0.5258115,  0.5187567;
            0.47743744, 0.52475524, 0.49378428, 0.52658814,  0.4994074,  0.4601859, 0.49075037,  0.5102478,  0.5258239,  0.5188128;
            0.47743654,  0.5247572,  0.4937819,  0.5265793, 0.49939516, 0.46018806, 0.49075553,  0.5102434,  0.5258276, 0.51882803;
            0.4774944,  0.5247596, 0.49380273,  0.5266573,   0.499469, 0.46016705, 0.49072716, 0.51027316,  0.5258121, 0.51875865;
            0.47743633,  0.5247571,  0.4937819, 0.52657926, 0.49939516, 0.46018812, 0.49075553,  0.5102434,  0.5258276,  0.5188279;
            0.47740635, 0.52475506, 0.49376696,  0.5265287, 0.49934444, 0.46020392,  0.4907762, 0.51022226,  0.5258391,  0.5188785;
            0.4774253, 0.52475554, 0.49377784, 0.52656543,  0.4993832,  0.4601925, 0.49076024,  0.5102382, 0.52582973,  0.5188381;
        ]
    ];

    (z, layers)
}