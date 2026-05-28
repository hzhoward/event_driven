"""
Options commission model for Backtrader.

Alpaca-style flat-rate: $0.65 per contract per leg.
Applied on every open AND close, so a 2-leg spread costs $1.30 to enter and
$1.30 to exit = $2.60 total round-trip per contract unit.

The slippage (bid/ask spread simulation) is handled separately inside
LongGammaStrategy via the iv_slip parameter (1 % of BSM theoretical price)
rather than here, so commission and slippage stay cleanly separated in the
P&L attribution.
"""
import backtrader as bt


class OptionsCommission(bt.CommInfoBase):
    """
    Fixed $0.65 fee per contract (= per unit of 100 shares).

    Backtrader calls _getcommission(size, price, pseudoexec) where:
      size  = number of contracts (signed: + buy, - sell)
      price = option price per share (not per contract)

    We charge per contract regardless of direction.
    """
    params = (
        ("commission", 0.65),   # $ per contract
        ("mult",       100.0),  # options multiplier (100 shares / contract)
        ("stocklike",  False),
        ("commtype",   bt.CommInfoBase.COMM_FIXED),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * self.p.commission
