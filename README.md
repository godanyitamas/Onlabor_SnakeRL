# Onlabor_SnakeRL
Megerősítéses tanulással működő Snake játék Catch-ről átírva

Snake class:

Négyféle mozgást tud a kígyó végrehajtani:
- a tényleges mozgás-koordináták a possible_actions array-ben vannak
- a neurháló 0,1,2,3 outputja adja a possible_actions indexét
- minden meghívásnál az update_state() azt is ellenőrzi, hogy nem akar-e saját maga fele menni a kígyó,
  ilyenkor egy random akciót választ


