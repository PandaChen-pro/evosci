/** Scores come from a single reviewer per idea in these runs, so the UI must not imply
 *  they are comparable across ideas or rounds. */
export function ScoreCaveat({ reviewers }: { reviewers: number }) {
  return (
    <div className="banner">
      <strong>这些分数该怎么读。</strong> 每个想法都是在它自己所属的那一轮里，由
      {reviewers === 1 ? "单个评审" : `${reviewers} 个评审`}打的分，轮次之间没有共同锚点。
      fitness 是某一份评审各项数值的加权和 —— 它可用于同一轮内排序，但不能拿一个想法的分数去和
      另一个想法比较，也不能当作跨轮次的走势线来读。凡是界面上出现的排名，都来自{" "}
      <code>report.md</code> 里的锦标赛结果，不是由这些数值算出来的。
    </div>
  );
}
