#include "include/int2048.h"

namespace sjtu {

class BigIntCore {
public:
  static const int base = 10000; // 1e4
  static const int base_digits = 4;
};

static void trim(std::vector<int> &a) {
  while (!a.empty() && a.back() == 0) a.pop_back();
}

static int cmp_abs(const std::vector<int> &a, const std::vector<int> &b) {
  if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
  for (int i = (int)a.size() - 1; i >= 0; --i) {
    if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
  }
  return 0;
}

static std::vector<int> add_abs(const std::vector<int> &a, const std::vector<int> &b) {
  const int base = BigIntCore::base;
  std::vector<int> res;
  res.reserve(std::max(a.size(), b.size()) + 1);
  int carry = 0;
  size_t n = std::max(a.size(), b.size());
  for (size_t i = 0; i < n || carry; ++i) {
    long long cur = carry;
    if (i < a.size()) cur += a[i];
    if (i < b.size()) cur += b[i];
    res.push_back((int)(cur % base));
    carry = (int)(cur / base);
  }
  return res;
}

static std::vector<int> sub_abs(const std::vector<int> &a, const std::vector<int> &b) {
  // assume |a| >= |b|
  const int base = BigIntCore::base;
  std::vector<int> res;
  res.reserve(a.size());
  int carry = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    long long cur = a[i] - carry - (i < b.size() ? b[i] : 0);
    if (cur < 0) { cur += base; carry = 1; } else carry = 0;
    res.push_back((int)cur);
  }
  trim(res);
  return res;
}

// FFT-based multiplication (Cooley-Tukey) using long double complex
struct C { long double x, y; C(long double X=0, long double Y=0):x(X),y(Y){} };
static C operator+(const C &a, const C &b){ return C(a.x+b.x, a.y+b.y);}
static C operator-(const C &a, const C &b){ return C(a.x-b.x, a.y-b.y);}
static C operator*(const C &a, const C &b){ return C(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);}

static void fft(std::vector<C> &a, bool invert) {
  int n = (int)a.size();
  // bit-reverse permutation
  for (int i=1, j=0; i<n; ++i) {
    int bit = n>>1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i<j) std::swap(a[i], a[j]);
  }
  for (int len=2; len<=n; len<<=1) {
    long double ang = 2.0L * 3.14159265358979323846264338327950288L / len * (invert ? -1.0L : 1.0L);
    C wlen(std::cos(ang), std::sin(ang));
    for (int i=0; i<n; i+=len) {
      C w(1,0);
      for (int j=0; j<len/2; ++j) {
        C u = a[i+j];
        C v = a[i+j+len/2] * w;
        a[i+j] = u + v;
        a[i+j+len/2] = u - v;
        w = w * wlen;
      }
    }
  }
  if (invert) {
    for (int i=0; i<n; ++i) { a[i].x /= n; a[i].y /= n; }
  }
}

static std::vector<int> mul_abs_fft(const std::vector<int> &a, const std::vector<int> &b) {
  // multiply arrays of base=1e4 digits
  std::vector<C> fa, fb;
  int n = 1;
  while (n < (int)a.size() + (int)b.size()) n <<= 1;
  fa.resize(n); fb.resize(n);
  for (size_t i=0;i<a.size();++i) fa[i].x = a[i];
  for (size_t i=a.size(); i<(size_t)n; ++i) fa[i] = C();
  for (size_t i=0;i<b.size();++i) fb[i].x = b[i];
  for (size_t i=b.size(); i<(size_t)n; ++i) fb[i] = C();
  fft(fa,false); fft(fb,false);
  for (int i=0;i<n;++i) fa[i] = fa[i]*fb[i];
  fft(fa,true);
  std::vector<int> res(n);
  long long carry = 0;
  const int base = BigIntCore::base;
  for (int i=0;i<n;++i) {
    long long t = (long long)std::llround(fa[i].x) + carry;
    res[i] = (int)(t % base);
    carry = t / base;
  }
  while (carry) { res.push_back((int)(carry % base)); carry /= base; }
  trim(res);
  return res;
}

class BigIntImpl {
public:
  bool neg = false;
  std::vector<int> d; // little-endian base 1e4

  BigIntImpl() {}
};

// Internal helpers bound to int2048
static BigIntImpl &impl_of(int2048 &x) { return *(BigIntImpl*)&x; }
static const BigIntImpl &impl_of(const int2048 &x) { return *(const BigIntImpl*)&x; }

// Constructors
int2048::int2048() {
  BigIntImpl &I = impl_of(*this); I.neg = false; I.d.clear();
}

int2048::int2048(long long v) {
  BigIntImpl &I = impl_of(*this);
  I.neg = (v < 0);
  unsigned long long u = I.neg ? (unsigned long long)(-(v + 1)) + 1ULL : (unsigned long long)v;
  const int base = BigIntCore::base;
  while (u) { I.d.push_back((int)(u % base)); u /= base; }
}

int2048::int2048(const std::string &s) { BigIntImpl &I = impl_of(*this); I.neg=false; I.d.clear(); read(s); }

int2048::int2048(const int2048 &o) {
  const BigIntImpl &O = impl_of(o);
  BigIntImpl &I = impl_of(*this);
  I.neg = O.neg; I.d = O.d;
}

void int2048::read(const std::string &s) {
  BigIntImpl &I = impl_of(*this);
  I.neg=false; I.d.clear();
  int pos = 0; while (pos<(int)s.size() && (s[pos]==' '||s[pos]=='\n' || s[pos]=='\t')) ++pos;
  if (pos<(int)s.size() && (s[pos]=='+'||s[pos]=='-')) { I.neg = (s[pos]=='-'); ++pos; }
  while (pos<(int)s.size() && s[pos]=='0') ++pos;
  std::vector<int> tmp;
  const int base = BigIntCore::base;
  const int bd = BigIntCore::base_digits;
  for (int i = (int)s.size(); i > pos; i -= bd) {
    int x = 0;
    int l = std::max(pos, i - bd);
    for (int j=l; j<i; ++j) x = x*10 + (s[j]-'0');
    tmp.push_back(x);
  }
  // tmp is big-endian chunks reversed; convert to little-endian
  I.d = tmp;
  trim(I.d);
  if (I.d.empty()) I.neg=false; // zero has no sign
}

void int2048::print() {
  const BigIntImpl &I = impl_of(*this);
  if (I.d.empty()) { std::cout << 0; return; }
  if (I.neg) std::cout << '-';
  int bd = BigIntCore::base_digits;
  std::cout << I.d.back();
  for (int i=(int)I.d.size()-2; i>=0; --i) {
    char buf[16]; std::snprintf(buf, sizeof(buf), "%0*d", bd, I.d[i]);
    std::cout << buf;
  }
}

int2048 &int2048::add(const int2048 &b) {
  BigIntImpl &A = impl_of(*this);
  const BigIntImpl &B = impl_of(b);
  if (A.neg == B.neg) {
    A.d = add_abs(A.d, B.d);
  } else {
    int c = cmp_abs(A.d, B.d);
    if (c >= 0) {
      A.d = sub_abs(A.d, B.d);
    } else {
      A.d = sub_abs(B.d, A.d);
      A.neg = B.neg;
    }
  }
  if (A.d.empty()) A.neg=false;
  return *this;
}

int2048 add(int2048 a, const int2048 &b) { a.add(b); return a; }

int2048 &int2048::minus(const int2048 &b) {
  BigIntImpl &A = impl_of(*this);
  const BigIntImpl &B = impl_of(b);
  if (A.neg != B.neg) {
    A.d = add_abs(A.d, B.d);
  } else {
    int c = cmp_abs(A.d, B.d);
    if (c >= 0) {
      A.d = sub_abs(A.d, B.d);
    } else {
      A.d = sub_abs(B.d, A.d);
      A.neg = !A.neg;
    }
  }
  if (A.d.empty()) A.neg=false;
  return *this;
}

int2048 minus(int2048 a, const int2048 &b) { a.minus(b); return a; }

int2048 int2048::operator+() const { return *this; }

int2048 int2048::operator-() const {
  int2048 r(*this);
  BigIntImpl &R = impl_of(r);
  if (!R.d.empty()) R.neg = !R.neg;
  return r;
}

int2048 &int2048::operator=(const int2048 &o) {
  const BigIntImpl &O = impl_of(o);
  BigIntImpl &I = impl_of(*this);
  if (this == &o) return *this;
  I.neg = O.neg; I.d = O.d; return *this;
}

int2048 &int2048::operator+=(const int2048 &b) { return add(b); }
int2048 operator+(int2048 a, const int2048 &b) { return a += b; }

int2048 &int2048::operator-=(const int2048 &b) { return minus(b); }
int2048 operator-(int2048 a, const int2048 &b) { return a -= b; }

int2048 &int2048::operator*=(const int2048 &b) {
  BigIntImpl &A = impl_of(*this);
  const BigIntImpl &B = impl_of(b);
  if (A.d.empty() || B.d.empty()) { A.d.clear(); A.neg=false; return *this; }
  // choose method
  std::vector<int> res;
  size_t n = A.d.size(), m = B.d.size();
  if (std::min(n,m) < 64) {
    const int base = BigIntCore::base;
    res.assign(n + m, 0);
    for (size_t i=0;i<n;++i) {
      long long carry = 0;
      for (size_t j=0;j<m || carry;++j) {
        long long cur = res[i+j] + (long long)A.d[i] * (j<m ? B.d[j] : 0) + carry;
        res[i+j] = (int)(cur % base);
        carry = cur / base;
      }
    }
    trim(res);
  } else {
    res = mul_abs_fft(A.d, B.d);
  }
  A.d.swap(res);
  A.neg = (A.neg != B.neg);
  if (A.d.empty()) A.neg=false;
  return *this;
}

int2048 operator*(int2048 a, const int2048 &b) { return a *= b; }

// Division and modulo (absolute) via base-digit long division with binary search
static int cmp_abs_be(const std::vector<int> &A, const std::vector<int> &B) {
  if (A.size() != B.size()) return A.size() < B.size() ? -1 : 1;
  for (size_t i=0;i<A.size();++i) {
    if (A[i] != B[i]) return A[i] < B[i] ? -1 : 1;
  }
  return 0;
}

static std::vector<int> mul_abs_be_digit(const std::vector<int> &B, int k) {
  const int base = BigIntCore::base;
  std::vector<int> R(B.size());
  long long carry = 0;
  for (int i=(int)B.size()-1; i>=0; --i) {
    long long cur = (long long)B[i]*k + carry;
    R[i] = (int)(cur % base);
    carry = cur / base;
  }
  if (carry) {
    R.insert(R.begin(), (int)carry);
  }
  // trim leading zeros be
  size_t p=0; while (p<R.size() && R[p]==0) ++p; if (p) R.erase(R.begin(), R.begin()+p);
  return R;
}

static std::vector<int> sub_abs_be(const std::vector<int> &A, const std::vector<int> &B) {
  // assume A >= B in big-endian base
  const int base = BigIntCore::base;
  int n = (int)std::max(A.size(), B.size());
  std::vector<int> R(n,0);
  // align to same length
  int ia = (int)A.size()-1, ib = (int)B.size()-1, ir = n-1;
  long long borrow = 0;
  for (; ir>=0; --ir) {
    long long av = (ia>=0 ? A[ia] : 0);
    long long bv = (ib>=0 ? B[ib] : 0);
    long long cur = av - bv - borrow;
    if (cur < 0) { cur += base; borrow = 1; } else borrow = 0;
    R[ir] = (int)cur;
    --ia; --ib;
  }
  // trim leading zeros
  size_t p=0; while (p<R.size() && R[p]==0) ++p; if (p) R.erase(R.begin(), R.begin()+p);
  return R;
}

static void div_mod_abs(const std::vector<int> &a_le, const std::vector<int> &b_le, std::vector<int> &q_le, std::vector<int> &r_le) {
  q_le.clear(); r_le.clear();
  if (b_le.empty()) return;
  if (cmp_abs(a_le, b_le) < 0) { r_le = a_le; return; }
  // convert to big-endian
  std::vector<int> A, B;
  A.assign(a_le.rbegin(), a_le.rend());
  B.assign(b_le.rbegin(), b_le.rend());
  std::vector<int> R; // remainder in big-endian
  std::vector<int> Q; Q.reserve(A.size());
  const int base = BigIntCore::base;
  for (size_t idx=0; idx<A.size(); ++idx) {
    // R = R * base + A[idx]
    if (!R.empty() || A[idx]!=0) {
      R.push_back(A[idx]);
    }
    if (R.empty()) { Q.push_back(0); continue; }
    // binary search qdigit in [0, base-1]
    int lo = 0, hi = base - 1, best = 0;
    while (lo <= hi) {
      int mid = (lo + hi) >> 1;
      std::vector<int> M = mul_abs_be_digit(B, mid);
      int cmp = cmp_abs_be(M, R);
      if (cmp <= 0) { best = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    if (best) {
      std::vector<int> M = mul_abs_be_digit(B, best);
      R = sub_abs_be(R, M);
    }
    // trim leading zeros in R
    size_t p=0; while (p<R.size() && R[p]==0) ++p; if (p) R.erase(R.begin(), R.begin()+p);
    Q.push_back(best);
  }
  // trim leading zeros in Q
  size_t p=0; while (p<Q.size() && Q[p]==0) ++p; if (p) Q.erase(Q.begin(), Q.begin()+p);
  // convert back to little-endian
  q_le.assign(Q.rbegin(), Q.rend()); trim(q_le);
  r_le.assign(R.rbegin(), R.rend()); trim(r_le);
}

int2048 &int2048::operator/=(const int2048 &b) {
  BigIntImpl &A = impl_of(*this);
  const BigIntImpl &B = impl_of(b);
  if (B.d.empty()) return *this; // undefined
  if (A.d.empty()) { A.neg=false; return *this; }
  std::vector<int> q, r;
  div_mod_abs(A.d, B.d, q, r);
  bool negQ = (A.neg != B.neg);
  if (!r.empty() && negQ) {
    // qabs++
    const int base = BigIntCore::base;
    int carry = 1;
    for (size_t i=0; i<q.size() && carry; ++i) {
      int cur = q[i] + carry;
      if (cur >= base) { q[i] = cur - base; carry = 1; } else { q[i] = cur; carry = 0; }
    }
    if (carry) q.push_back(carry);
  }
  A.d.swap(q);
  A.neg = (!A.d.empty() ? negQ : false);
  return *this;
}

int2048 operator/(int2048 a, const int2048 &b) { return a /= b; }

int2048 &int2048::operator%=(const int2048 &b) {
  BigIntImpl &A = impl_of(*this);
  const BigIntImpl &B = impl_of(b);
  if (B.d.empty()) return *this; // undefined
  if (A.d.empty()) { A.neg=false; return *this; }
  std::vector<int> q, r;
  div_mod_abs(A.d, B.d, q, r);
  if (!r.empty() && (A.neg != B.neg)) {
    r = add_abs(r, B.d);
  }
  A.d.swap(r);
  A.neg = (!A.d.empty() ? B.neg : false);
  return *this;
}

int2048 operator%(int2048 a, const int2048 &b) { return a %= b; }

std::istream &operator>>(std::istream &in, int2048 &x) {
  std::string s; in >> s; x.read(s); return in;
}

std::ostream &operator<<(std::ostream &out, const int2048 &x) {
  const BigIntImpl &I = impl_of(x);
  if (I.d.empty()) { out << 0; return out; }
  if (I.neg) out << '-';
  int bd = BigIntCore::base_digits;
  out << I.d.back();
  for (int i=(int)I.d.size()-2; i>=0; --i) {
    char buf[16]; std::snprintf(buf, sizeof(buf), "%0*d", bd, I.d[i]);
    out << buf;
  }
  return out;
}

bool operator==(const int2048 &a, const int2048 &b) {
  const BigIntImpl &A = impl_of(a); const BigIntImpl &B = impl_of(b);
  return A.neg==B.neg && A.d==B.d;
}

bool operator!=(const int2048 &a, const int2048 &b) { return !(a==b); }

bool operator<(const int2048 &a, const int2048 &b) {
  const BigIntImpl &A = impl_of(a); const BigIntImpl &B = impl_of(b);
  if (A.neg != B.neg) return A.neg;
  int c = cmp_abs(A.d, B.d);
  return A.neg ? (c>0) : (c<0);
}

bool operator>(const int2048 &a, const int2048 &b) { return b < a; }
bool operator<=(const int2048 &a, const int2048 &b) { return !(b<a); }
bool operator>=(const int2048 &a, const int2048 &b) { return !(a<b); }

} // namespace sjtu
