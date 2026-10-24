# Message to the student (casual version)

Copy and send as is. Attach `evaluation/confusion_matrix_test.png` and
`evaluation/training_curves.png`.

---

Hi po! Eto na po yung results ng model natin 😊

Bale nag-test po kami gamit yung 50 images na hindi pa nakikita ng model
habang tinetrain siya, para fair po yung result. Nakakuha po siya ng
**92% accuracy**.

Eto po yung breakdown per class:

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Immature | 0.82 | 0.93 | 0.88 |
| Mature | 0.93 | 0.82 | 0.88 |
| Overmature | 1.00 | 1.00 | 1.00 |
| **Average** | **0.92** | **0.92** | **0.92** |

Yung overmature po, perfect, wala siyang mali. Yung konting mali po ay
nasa pagitan ng immature at mature, which makes sense kasi magkahawig
talaga sila tingnan.

Para lang din po sure kami, inulit namin yung test nang ilang beses gamit
ibang set ng images, and nasa **92% to 100%** po yung lumalabas, mga 95%
on average. So consistent naman po siya.

Kung kailangan niyo po sa paper:
- **Dataset:** 333 images (101 immature, 112 mature, 120 overmature)
- **Split:** 70% training, 15% validation, 15% testing
- **Model:** CNN (Convolutional Neural Network)

Attached ko na rin po yung confusion matrix at yung training graph. Sabihan
niyo lang po kami kung may kailangan pa kayong idagdag! 🙏

---

## Kung magtanong sila kung ano ibig sabihin

- **Accuracy:** Ilan sa lahat ng images ang tama yung hula.
- **Precision:** Kapag sinabi ng model na "mature", gaano kadalas na
  mature talaga.
- **Recall:** Sa lahat ng totoong mature, ilan ang nahuli ng model.
- **F1-score:** Pinagsamang score ng precision at recall, para isang
  number na lang per class.
